import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from transformers import TFAutoModelForSemanticSegmentation
import cv2

# -----------------------
# CONFIG
# -----------------------
fold_dir = Path("/path/to/tmeseg_cv") #download from zenodo

img_root  = Path("/path/to/image") #download from zenodo
mask_root = Path("/path/to/maskPng") #download from zenodo
out_dir = Path("/path/to/tmeseg_cv/pre_mask")

image_size = 512
batch_size = 8
auto = tf.data.AUTOTUNE
n_splits = 5

id2label = {
    0: "background",
    1: "tumor",
    2: "stroma",
    3: "benign_epi",
    4: "necrosis_hemo",
    5: "adipose",
}
num_labels = len(id2label)
BG = 0
FG_CLASSES = list(range(1, num_labels))

#openCV: BGR
class_colors_artemis = [(0, 0, 0), (0, 0, 128), (0, 204, 255), (255, 255, 0),(255, 0, 255),(0, 128, 128)]

# Point these to your saved fold model folders (edit paths if needed)
# Example: model_dirs[1] is the folder created by save_pretrained for fold 1
model_dirs = {
    1: Path("/path/to/tmeseg-infer/model/mit-b3-finetuned-TCGAbcss-biopsy-fold1"),
    2: Path("/path/to/tmeseg-infer/model/mit-b3-finetuned-TCGAbcss-biopsy-fold2"),
    3: Path("/path/to/tmeseg-infer/model/mit-b3-finetuned-TCGAbcss-biopsy-fold3"),
    4: Path("/path/to/tmeseg-infer/model/mit-b3-finetuned-TCGAbcss-biopsy-fold4"),
    5: Path("/path/to/tmeseg-infer/model/mit-b3-finetuned-TCGAbcss-biopsy-fold5"),
}

# -----------------------
# IO + DATASET
# -----------------------
def make_image_mask_paths(file_names):
    stems = [Path(fn).stem for fn in file_names]
    imgs  = [str(img_root / f"{s}.png") for s in stems]
    masks = [str(mask_root / f"mask_{s}.png") for s in stems]
    return imgs, masks, stems

def read_image(path):
    x = tf.io.read_file(path)
    x = tf.image.decode_png(x, channels=3)
    x = tf.image.resize(x, [image_size, image_size])
    x = tf.cast(x, tf.float32) / 255.0
    # channel-first to match your training
    x = tf.transpose(x, (2, 0, 1))  # (3,H,W)
    return x

def read_mask(path):
    y = tf.io.read_file(path)
    y = tf.image.decode_png(y, channels=1)
    y = tf.image.resize(y, [image_size, image_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    y = tf.cast(y, tf.int32)
    y = tf.squeeze(y, axis=-1)  # (H,W)
    return y

def preprocess(img_path, mask_path):
    return read_image(img_path), read_mask(mask_path)

def build_test_ds(img_paths, mask_paths):
    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    ds = ds.map(preprocess, num_parallel_calls=auto)
    ds = ds.batch(batch_size).prefetch(auto)
    return ds

# -----------------------
# PREDICTION (UPSAMPLE LOGITS -> ARGMAX)
# -----------------------
def predict_mask(model, xb, target_hw=(512, 512)):
    """
    xb: (B,3,H,W)
    returns: (B,H,W) int32 at target size
    """
    out = model(xb, training=False)
    logits = out.logits  # expected (B, C, h, w)

    # to channel-last for resize: (B,h,w,C)
    logits = tf.transpose(logits, (0, 2, 3, 1))
    logits_up = tf.image.resize(logits, target_hw, method="bilinear")
    pred = tf.argmax(logits_up, axis=-1)  # (B,H,W)
    return pred.numpy().astype(np.int32)

# -----------------------
# METRICS FROM CONFUSION MATRIX
# -----------------------
def fast_confusion(y_true, y_pred, n_classes):
    """
    y_true,y_pred: (H,W) int
    returns cm: (n_classes,n_classes) where rows=true, cols=pred
    """
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    m = (yt >= 0) & (yt < n_classes)
    yt = yt[m]
    yp = yp[m]
    return np.bincount(n_classes * yt + yp, minlength=n_classes**2).reshape(n_classes, n_classes)

def dice_from_cm(cm, eps=1e-7):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    return (2*tp + eps) / (2*tp + fp + fn + eps)

def recall_from_cm(cm, eps=1e-7):
    tp = np.diag(cm).astype(np.float64)
    fn = cm.sum(axis=1).astype(np.float64) - tp
    return (tp + eps) / (tp + fn + eps)

def npv_from_cm(cm, eps=1e-7):
    """
    One-vs-rest NPV per class:
      NPV_c = TN_c / (TN_c + FN_c)
    """
    total = cm.sum().astype(np.float64)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    tn = total - (tp + fp + fn)
    return (tn + eps) / (tn + fn + eps)

def macro_excluding_bg(vec, classes, ignore_nan=True):
    vals = vec[classes]
    if ignore_nan:
        vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if len(vals) else float("nan")

# -----------------------
# MAIN: EVAL 5 FOLDS
# -----------------------
summary_rows = []

for fold in range(1, n_splits + 1):
    fold_csv = fold_dir / f"fold_{fold}.csv"
    mask_out_path = out_dir/f"fold_{fold}"
    mask_out_path.mkdir(parents=True, exist_ok=True)

    if not fold_csv.exists():
        raise FileNotFoundError(f"Missing fold file: {fold_csv}")
    if fold not in model_dirs or not model_dirs[fold].exists():
        raise FileNotFoundError(f"Missing model dir for fold {fold}: {model_dirs.get(fold)}")

    df_test = pd.read_csv(fold_csv)
    if "file_name" not in df_test.columns:
        raise ValueError(f"{fold_csv} must contain 'file_name' column")

    test_imgs, test_masks, test_stems = make_image_mask_paths(df_test["file_name"].tolist())
    test_ds = build_test_ds(test_imgs, test_masks)

    # Load model (no training)
    model = TFAutoModelForSemanticSegmentation.from_pretrained(str(model_dirs[fold]))

    # Fold aggregation
    fold_cm = np.zeros((num_labels, num_labels), dtype=np.int64)
    per_patch_records = []

    idx = 0
    for xb, yb in test_ds:
        preds = predict_mask(model, xb, target_hw=(image_size, image_size))
        ytrue = yb.numpy().astype(np.int32)

        # BGR color table (index = class id)
        color_lut = np.array(class_colors_artemis, dtype=np.uint8)  # (C,3)

        B = preds.shape[0]
        for b in range(B):
            if idx + b >= len(test_stems):
                break

            # preds[b]: (H,W) int -> (H,W,3) BGR
            colored = color_lut[preds[b]]  # numpy advanced indexing

            # output name follows your existing stem logic
            out_name = f"{test_stems[idx + b]}.png"
            out_path = mask_out_path / out_name

            # cv2 expects BGR uint8 — already satisfied
            cv2.imwrite(str(out_path), colored)


        B = preds.shape[0]
        for b in range(B):
            if idx >= len(test_stems):
                break

            cm = fast_confusion(ytrue[b], preds[b], num_labels)
            fold_cm += cm

            dice_vec = dice_from_cm(cm)
            rec_vec  = recall_from_cm(cm)
            npv_vec  = npv_from_cm(cm)

            # Balanced accuracy (multiclass) = macro recall over foreground classes
            bal_acc_no_bg = macro_excluding_bg(rec_vec, FG_CLASSES)

            # Macro NPV excluding background (one-vs-rest per class)
            npv_macro_no_bg = macro_excluding_bg(npv_vec, FG_CLASSES)

            rec = {
                "fold": fold,
                "file_name": df_test["file_name"].iloc[idx],
                "balanced_acc_no_bg": bal_acc_no_bg,
                "npv_macro_no_bg": npv_macro_no_bg,
                "dice_mean_all": float(np.mean(dice_vec)),
                "dice_mean_no_bg": macro_excluding_bg(dice_vec, FG_CLASSES),
            }
            for c in range(num_labels):
                rec[f"dice_{c}_{id2label[c]}"] = float(dice_vec[c])
                rec[f"npv_{c}_{id2label[c]}"]  = float(npv_vec[c])
                rec[f"recall_{c}_{id2label[c]}"] = float(rec_vec[c])

            per_patch_records.append(rec)
            idx += 1

        if idx >= len(test_stems):
            break

    # Save per-patch metrics
    per_patch_df = pd.DataFrame(per_patch_records)
    out_patch_csv = fold_dir / f"test_patch_metrics_fold{fold}.csv"
    #per_patch_df.to_csv(out_patch_csv, index=False)

    # Save fold confusion matrix
    cm_df = pd.DataFrame(
        fold_cm,
        index=[f"true_{i}_{id2label[i]}" for i in range(num_labels)],
        columns=[f"pred_{i}_{id2label[i]}" for i in range(num_labels)],
    )
    out_cm_csv = fold_dir / f"confusion_matrix_fold{fold}.csv"
    cm_df.to_csv(out_cm_csv)

