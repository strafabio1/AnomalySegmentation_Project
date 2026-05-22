# Anomaly Evaluation Suite

This directory contains the unified pipeline for evaluating anomaly segmentation and Out-Of-Distribution (OOD) detection performance. It is designed to work seamlessly with both the **ERFNet** baseline and the **EoMT** model.

## Requirements

Ensure you have installed the project's main dependencies before running the evaluation scripts. You can install them from the repository root:

```bash
pip install -r requirements.txt
```

## Directory Structure

- **`evalAnomaly.py`**: The main evaluation script that computes anomaly scores (MSP, MaxLogit, MaxEntropy, and optionally RbA for EoMT) and evaluation metrics (AUPRC and FPR@TPR95).
- **`model_builder.py`**: Helper module to load ERFNet and EoMT checkpoints and configurations.
- **`ood_metrics.py`**: Contains the logic to compute standard OOD metrics (AUPRC, FPR@TPR95) from prediction scores and ground truth masks.
- **`post_hoc.py`**: Implementation of several post-hoc scoring functions (MSP, MaxLogit, MaxEntropy, RbA) adapted for both architectures.

## Datasets

For testing the anomaly segmentation model, the supported datasets include Road Anomaly, Road Obstacle, Fishyscapes, LostAndFound, and Streethazard. 
You can download the testing images [here](https://drive.google.com/file/d/1r2eFANvSlcUjxcerjC8l6dRa0slowMpx/view) or the validation datasets [here](https://drive.google.com/file/d/1zcayoIIJztxKuHOIjmSjGoQBDy4RdETr/view?usp=drive_link).

## Anomaly Evaluation Pipeline

**⚠️ IMPORTANT: All commands must be executed from the root directory of the repository.**

The `evalAnomaly.py` script automatically computes and reports the metrics on common anomaly segmentation datasets. The script expects the dataset to follow a standard structure where `images` and `labels_masks` folders are present.

### Evaluating ERFNet

To evaluate the ERFNet model, provide the path to the images and the checkpoint weights:

```bash
python -m anomaly_evaluation.evalAnomaly \
  --model_type erfnet \
  --input '/path/to/dataset/images/*.png' \
  --weights /path/to/erfnet_pretrained.pth
```

### Evaluating EoMT

To evaluate the EoMT model, you must also provide the corresponding configuration file:

```bash
python -m anomaly_evaluation.evalAnomaly \
  --model_type eomt \
  --input '/path/to/dataset/images/*.png' \
  --weights /path/to/eomt_checkpoint.bin \
  --config /path/to/config.yaml
```

### Advanced Options

- **Temperature Scaling**: You can evaluate the effect of temperature scaling on the calibration of the anomaly scores by providing one or more temperature values using the `--temperature` argument:
  ```bash
  python -m anomaly_evaluation.evalAnomaly \
    --model_type erfnet \
    --input '/path/to/dataset/images/*.png' \
    --weights /path/to/erfnet_pretrained.pth \
    --temperature 1.0 1.5 2.0
  ```

## Notes

- The script automatically handles target image sizes: ERFNet uses `(512, 1024)`, while EoMT dynamically adjusts depending on whether it was trained on Cityscapes `(1024, 1024)` or COCO `(640, 640)`.
- The evaluation requires anomalous test images and ground truth masks formatted appropriately for the datasets specified.
