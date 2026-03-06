## TMEseg
This repository is for the manuscript "Al-Derived Tumor-Infiltrating Lymphocytes Enhance the Prediction of Pathologic Complete Response in an Early-Stage Triple-Negative Breast Cancer Prospective Trial" submitted to Clinical Cancer Research. It could guide you to generate TME masks with a well-trained AI model for semantic segmentation.

### Generating tiles for whole slide images
Please follow instructions in https://github.com/xi11/AIgrading/generating_tiles.

### Building env with Dockerfile

### Training
```
cd ./tmeseg-train
/usr/bin/python3 segformer_tmeseg_ftartemis.py
```

### Inference
```
cd ./tmeseg-infer
/usr/bin/python3 main_tme.py \
    -d /path/to/1_cws_tiling \
    -o /path/to/mit-b3-finetunedBRCA/mask_cws512 \
    -s /path/to/mit-b3-finetunedBRCA/mask_ss1512 \
    -sp /path/to/mit-b3-finetunedBRCA/mask_ss1512_post \
    -p "*.svs" \
    -c \
    -ps 512 \
    -ins 512 \
    -nC 6 \
    -n {{index}} \
    -sf 0.0625 \
    -nJ 32

```
Users can also have a reproducible run through [Code Ocean](https://codeocean.com/capsule/7777436/tree/v1).
