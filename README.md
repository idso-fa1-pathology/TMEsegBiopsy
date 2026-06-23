## TMEseg
This repository is for the manuscript "AI-Derived Tumor-Infiltrating Lymphocytes Enhance the Prediction of Pathologic Complete Response in Early-Stage Triple-Negative Breast Cancer". It could guide you to generate TME masks with a well-trained AI model for semantic segmentation.

### Generating tiles for whole slide images
Please follow instructions in https://github.com/xi11/AIgrading/generating_tiles.

### Building env with Dockerfile

### Training
```
cd ./tmeseg-train
/usr/bin/python3 segformer_tmeseg_biopsy.py
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
