## TMEseg
This repository is for the manuscript "AI-Derived Tumor-Infiltrating Lymphocytes Enhance the Prediction of Pathologic Complete Response in Early-Stage Triple-Negative Breast Cancer". It could guide you to generate TME masks with a well-trained AI model for semantic segmentation.

### 1. Training data

Download the training set from Zenodo: <https://zenodo.org/records/18407664>

The training scripts in `tmeseg-train` expect the extracted data as an `image/` folder of PNG patches and a matching `maskPng/` folder of `mask_<name>.png` label patches, plus a `tmeseg_cv/` folder with `fold_1.csv` ... `fold_5.csv` (each with a `file_name` column) for the 5-fold CV split. Update `image_dir`, `mask_dir`, and `splits_dir` in the scripts (see step 5/6) to point at wherever you extract these.

### 2. Clone this repository

```         
git clone https://github.com/idso-fa1-pathology/TMEsegBiopsy.git
cd TMEsegBiopsy
```

The steps below assume commands are run from inside this `TMEsegBiopsy` folder unless stated otherwise.

### 3. Pretrained checkpoint

Download `checkpoint9class` from <https://huggingface.co/idso-fa1-pathology/TMEsegBreastTCGA/tree/main>:

```         
pip install -U huggingface_hub
```

``` python
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download

repo_id = "idso-fa1-pathology/TMEsegBreastTCGA"
subfolder = "checkpoint9class"
# IMPORTANT: save it under this exact name/path -- it's what tmeseg-train
# scripts hard-code as `model_checkpoint`. 
output_dir = Path("./tmeseg-infer/model/mit-b3-finetuned-TCGAbcss9class")
output_dir.mkdir(parents=True, exist_ok=True)

for filename in ["config.json", "tf_model.h5"]:
    downloaded_file = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder, repo_type="model")
    shutil.copy(downloaded_file, output_dir / filename)
```

### 4. Building the environment

```         
docker build -t tmeseg .
```

Mount this repo into the container at `/App` (the image's `WORKDIR`), and pass `--gpus all` so TensorFlow/PyTorch can see the GPU. You'll also need a mount for anything the container needs to read or write that lives outside the repo -- the extracted training set (step 1), the raw slides you want to run inference on, and an output directory. Because the container is run with `--rm`, anything written to a path that isn't mounted is lost the moment the container exits, so the output mount in particular is required, not optional:

```         
docker run --gpus all -it --rm \
    -v /path/to/TMEsegBiopsy:/App \
    -v /path/to/tme_annotations:/data \
    -v /path/to/raw_slides:/slides \
    -v /path/to/output:/output \
    tmeseg
```

This drops you into a bash shell inside the container at `/App`, with the repo's `tmeseg-train`, `tmeseg-eval`, and `tmeseg-infer` folders directly available. The pretrained/fine-tuned checkpoints (step 3, and the models saved by steps 5-6) live inside the repo itself, so they come along for free with the `/App` mount -- no separate mount needed for those. From there, run any of the training/CV/inference commands in steps 5-7 exactly as written (e.g. `cd tmeseg-train && /usr/bin/python3 segformer_tmeseg_biopsy.py`), using `/data/...` for training data, `/slides` for the inference input in step 7's `-d`, and `/output` for its `-o`.

### 5. Reproducing the 5-fold CV results

These two scripts take no command-line arguments -- open each one and edit the paths in the `if __name__ == "__main__":` block at the bottom before running.

Train the 5 folds:

```         
cd ./tmeseg-train
```

Edit `image_dir`, `mask_dir`, `splits_dir` (all from step 1) in `segformer_tmesegCV_biopsy.py`, then run:

```         
/usr/bin/python3 segformer_tmesegCV_biopsy.py
```

This saves one model per fold to `tmeseg-infer/model/mit-b3-finetuned-TCGAbcss-biopsy-fold1` ... `-fold5`.

Evaluate the 5 folds:

```         
cd ../tmeseg-eval
```

Edit `fold_dir`, `img_root`, `mask_root`, `out_dir`, and `model_dirs` at the top of `evaluate5CV_balanced_acc_npy.py` (the `model_dirs` paths should already match the fold checkpoints saved above), then run:

```         
/usr/bin/python3 evaluate5CV_balanced_acc_npy.py
```

This writes stitched prediction masks to `out_dir` and per-fold confusion matrices to `<fold_dir>/confusion_matrix_fold1.csv` ... `fold5.csv`.

### 6. Training the final fine-tuned model

```         
cd ./tmeseg-train
```

Edit `image_dir` and `mask_dir` (from step 1) in `segformer_tmeseg_biopsy.py`, then run:

```         
/usr/bin/python3 segformer_tmeseg_biopsy.py
```

This saves the final model to `tmeseg-infer/model/mit-b3-finetuned-TCGAbcss-biopsy`, which is the checkpoint `tmeseg-infer/predict_slide.py` loads for inference.

### 7. Inference

```         
cd ./tmeseg-infer
/usr/bin/python3 main_tme.py \
    -d /slides \
    -o /output \
    -p "*.svs" \
    -c \
    -ps 512 \
    -ins 512 \
    -nC 6 \
    -n 0 \
    -sf 0.0625 \
    -nJ 5
```

`-d`/`-o` above are the `/slides` and `/output` container paths mounted in step 4 -- swap in whatever you mounted your raw slides and output directory to.

Tiling code lives in `tmeseg-infer/cws_tiling/` (`main_tiles.py`, `save_cws.py`, `cws_generator.py`); `main_tme.py` imports it directly, so no external tiling repo needs to be cloned or mounted.

Output layout under `-o/--save_dir`:

```         
<save_dir>/cws_tiling/          tiles
<save_dir>/mask_cws512/         per-tile TME prediction
<save_dir>/mask_ss1512/         stitched slide-level mask
<save_dir>/mask_ss1512_post/    stitched mask after post-processing
```

### 8. Color code of the output masks

Each pixel in the output masks is colored by its predicted class (the `argmax` of the model's per-class logits):

| Class ID | Color (RGB)   | Tissue type         |
|----------|---------------|---------------------|
| 0        | (0, 0, 0)     | background          |
| 1        | (128, 0, 0)   | tumor               |
| 2        | (255, 204, 0) | stroma              |
| 3        | (0, 255, 255) | benign_epithelium   |
| 4        | (255, 0, 255) | necrosis_hemorrhage |
| 5        | (128, 128, 0) | adipose_tissue      |

These are the RGB values as they actually appear when the mask files are opened (e.g. `mask_ss1512_post/*.png`), from `tmeseg-infer/ss1_post.py`'s `tme_colors` (classes 1-5) plus background. `tmeseg-infer/predict_slide.py` defines the same palette as `class_colors_artemis`, but stores it as OpenCV BGR tuples -- e.g. tumor is `(0, 0, 128)` there, which is the same color as `(128, 0, 0)` RGB above, just with the first and third channels swapped. If you're pulling colors from `class_colors_artemis` directly, reverse each tuple to get the RGB value a normal image viewer will show.

