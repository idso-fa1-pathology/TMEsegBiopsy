import os
import random as rn
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam
from transformers import TFAutoModelForSemanticSegmentation


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    rn.seed(seed)


def read_image(img_path, image_size):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    return image


def read_mask(mask_path, image_size):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)

    mask = tf.image.resize(
        mask,
        [image_size, image_size],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )

    return mask


def aug_transforms(image):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.random_brightness(image, 0.25)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    image = tf.image.random_saturation(image, 0.8, 2.0)
    image = tf.image.random_hue(image, 0.1)
    return image


def load_img_train(img_path, mask_path, image_size):
    image = read_image(img_path, image_size)
    mask = read_mask(mask_path, image_size)

    image = aug_transforms(image)

    # SegFormer TF input: C x H x W
    image = tf.transpose(image, (2, 0, 1))

    # Mask: H x W integer class IDs
    mask = tf.squeeze(mask, axis=-1)
    mask = tf.cast(mask, tf.int32)

    return image, mask


def build_dataset(image_paths, mask_paths, image_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    dataset = dataset.map(
        lambda img, mask: load_img_train(img, mask, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def get_all_image_and_mask_paths(image_dir, mask_dir):
    image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
    image_names = [Path(path).stem for path in image_paths]

    mask_paths = [
        os.path.join(mask_dir, f"mask_{name}.png")
        for name in image_names
    ]

    return image_paths, mask_paths, image_names


def read_test_names_from_fold_csv(fold_csv):
    """
    Reads test image names from a fold CSV.

    Expected CSV format:
        file_name
        xxx.png
        yyy.png

    Returns image stems without .png.
    """
    df = pd.read_csv(fold_csv)

    if "file_name" not in df.columns:
        raise ValueError(
            f"'file_name' column not found in {fold_csv}. "
            f"Columns are: {list(df.columns)}"
        )

    test_names = set(
        Path(x).stem
        for x in df["file_name"].dropna().astype(str)
    )

    return test_names


def get_train_paths_for_fold(image_paths, mask_paths, image_names, fold_csv):
    test_names = read_test_names_from_fold_csv(fold_csv)
    all_names = set(image_names)

    missing_test_names = test_names - all_names
    if missing_test_names:
        print(
            f"[WARNING] {len(missing_test_names)} test images from {fold_csv} "
            f"were not found in image_dir."
        )
        print("First few missing test names:")
        for name in sorted(list(missing_test_names))[:10]:
            print(name)

    train_image_paths = []
    train_mask_paths = []
    train_image_names = []

    for image_path, mask_path, image_name in zip(image_paths, mask_paths, image_names):
        if image_name not in test_names:
            train_image_paths.append(image_path)
            train_mask_paths.append(mask_path)
            train_image_names.append(image_name)

    return train_image_paths, train_mask_paths, train_image_names, test_names


def check_mask_paths(mask_paths):
    missing_masks = [path for path in mask_paths if not os.path.exists(path)]

    if missing_masks:
        print(f"[ERROR] Missing {len(missing_masks)} mask files.")
        print("First few missing masks:")
        for path in missing_masks[:10]:
            print(path)

        raise FileNotFoundError("Some mask files are missing.")


def train_one_fold(
    fold_idx,
    fold_csv,
    all_image_paths,
    all_mask_paths,
    all_image_names,
    image_size,
    learning_rate,
    batch_size,
    num_epochs,
    model_checkpoint,
    output_dir,
    seed,
    id2label,
    label2id,
):
    print("=" * 80)
    print(f"Training fold {fold_idx}")
    print(f"Fold CSV: {fold_csv}")
    print("=" * 80)

    set_seed(seed + fold_idx)

    train_image_paths, train_mask_paths, train_image_names, test_names = (
        get_train_paths_for_fold(
            image_paths=all_image_paths,
            mask_paths=all_mask_paths,
            image_names=all_image_names,
            fold_csv=fold_csv,
        )
    )

    num_total = len(all_image_paths)
    num_train = len(train_image_paths)
    num_test_found = num_total - num_train
    num_test_csv = len(test_names)

    print(f"Number of total images: {num_total}")
    print(f"Number of test images listed in CSV: {num_test_csv}")
    print(f"Number of test images found in image_dir: {num_test_found}")
    print(f"Number of training images for fold {fold_idx}: {num_train}")

    if num_train == 0:
        raise ValueError(f"No training images found for fold {fold_idx}")

    check_mask_paths(train_mask_paths)

    train_ds = build_dataset(
        image_paths=train_image_paths,
        mask_paths=train_mask_paths,
        image_size=image_size,
        batch_size=batch_size,
    )

    optimizer = Adam(learning_rate=learning_rate)

    model = TFAutoModelForSemanticSegmentation.from_pretrained(
        model_checkpoint,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    model.compile(optimizer=optimizer)

    steps_per_epoch = max(1, num_train // batch_size)

    model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        verbose=1,
    )

    model_name = Path(model_checkpoint).name
    model_id = os.path.join(
        output_dir,
        f"{model_name}-biopsy-fold{fold_idx}",
    )

    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(model_id)
    print(f"Fold {fold_idx} model saved to: {model_id}")

    tf.keras.backend.clear_session()


def main(
    image_dir,
    mask_dir,
    splits_dir,
    image_size,
    learning_rate,
    batch_size,
    num_epochs,
    model_checkpoint,
    output_dir,
    seed,
    id2label,
    label2id,
):
    os.makedirs(output_dir, exist_ok=True)

    all_image_paths, all_mask_paths, all_image_names = get_all_image_and_mask_paths(
        image_dir=image_dir,
        mask_dir=mask_dir,
    )

    num_img = len(all_image_paths)
    print(f"Number of all images: {num_img}")

    if num_img == 0:
        raise ValueError(f"No PNG images found in: {image_dir}")

    check_mask_paths(all_mask_paths)

    for fold_idx in range(1, 6):
        fold_csv = os.path.join(splits_dir, f"fold_{fold_idx}.csv")

        if not os.path.exists(fold_csv):
            raise FileNotFoundError(f"Fold CSV not found: {fold_csv}")

        train_one_fold(
            fold_idx=fold_idx,
            fold_csv=fold_csv,
            all_image_paths=all_image_paths,
            all_mask_paths=all_mask_paths,
            all_image_names=all_image_names,
            image_size=image_size,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
            model_checkpoint=model_checkpoint,
            output_dir=output_dir,
            seed=seed,
            id2label=id2label,
            label2id=label2id,
        )


if __name__ == "__main__":

    # -----------------------
    # Hyperparameter settings
    # -----------------------
    seed = 2023

    image_dir = "/path_to_tme_annotations/image"
    mask_dir = "/path_to_tme_annotations/maskPng"
    splits_dir = "/path_to_tme_annotations/tmeseg_cv"

    image_size = 512
    learning_rate = 0.0001
    batch_size = 8
    num_epochs = 60

    model_checkpoint = "../tmeseg-infer/model/mit-b3-finetuned-TCGAbcss"
    output_dir = "../tmeseg-infer/model"

    # -----------------------
    # Label settings
    # -----------------------
    id2label = {
        0: "background",
        1: "tumor",
        2: "stroma",
        3: "benign_epi",
        4: "necrosis_hemo",
        5: "adipose",
    }

    label2id = {name: idx for idx, name in id2label.items()}

    main(
        image_dir=image_dir,
        mask_dir=mask_dir,
        splits_dir=splits_dir,
        image_size=image_size,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        model_checkpoint=model_checkpoint,
        output_dir=output_dir,
        seed=seed,
        id2label=id2label,
        label2id=label2id,
    )