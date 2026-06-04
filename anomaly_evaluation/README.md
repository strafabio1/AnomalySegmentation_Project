# Anomaly Evaluation

This directory contains the unified pipeline for evaluating anomaly segmentation and Out-Of-Distribution (OOD) detection performance. It is designed to work seamlessly with both the **ERFNet** baseline and the **EoMT** model.

## Requirements

Ensure you have installed the project's main dependencies before running the evaluation scripts. You can install them from the repository root:

```bash
pip install -r requirements.txt
```

## Directory Structure

- **`compare_models.py`**: Script to qualitatively compare the anomaly maps produced by ERFNet, EoMT (Cityscapes), EoMT (COCO), and the fine-tuned EoMT using a single post-hoc method.
- **`evalAnomaly.py`**: The main evaluation script that computes anomaly scores (MSP, MaxLogit, MaxEntropy, and RbA for EoMT) and evaluation metrics (AUPRC and FPR@TPR95).
- **`model_builder.py`**: Helper module to load ERFNet and EoMT checkpoints and configurations.
- **`ood_metrics.py`**: Contains the logic to compute standard OOD metrics (AUPRC, FPR@TPR95) from prediction scores and ground truth masks.
- **`post_hoc.py`**: Implementation of several post-hoc scoring functions (MSP, MaxLogit, MaxEntropy, RbA) adapted for both architectures.

## Datasets

For testing the anomaly segmentation model, the supported datasets include Road Anomaly, Road Obstacle, Fishyscapes, LostAndFound, and Streethazard.
You can download the validation datasets [here](https://drive.google.com/file/d/1zcayoIIJztxKuHOIjmSjGoQBDy4RdETr/view?usp=drive_link).

## Anomaly Evaluation Pipeline

**⚠️ IMPORTANT: All commands must be executed from the root directory of the repository.**

The `evalAnomaly.py` script automatically computes and reports the metrics on common anomaly segmentation datasets. The script expects the dataset to follow a standard structure where `images` and `labels_masks` folders are present.

### Evaluating ERFNet

To evaluate the ERFNet model, provide the path to the images and the checkpoint weights:

```bash
python -m anomaly_evaluation.evalAnomaly \
  --model_type erfnet \
  --input './path/to/dataset/images/*.png' \
  --weights './path/to/erfnet_pretrained.pth'
```

🔧 Replace `./path/to/dataset/images/*.png` with the actual path to your dataset images.
🔧 Replace `./path/to/erfnet_pretrained.pth` with the path to your ERFNet checkpoint.

### Evaluating EoMT

To evaluate the EoMT model, you must also provide the corresponding configuration file:

```bash
python -m anomaly_evaluation.evalAnomaly \
  --model_type eomt \
  --input './path/to/dataset/images/*.png' \
  --weights /path/to/eomt_checkpoint.bin \
  --config './path/to/config.yaml'
```

🔧 Replace `./path/to/dataset/images/*.png` with the actual path to your dataset images.
🔧 Replace `/path/to/eomt_checkpoint.bin` with the path to your EoMT checkpoint.
🔧 Replace `./path/to/config.yaml` with the path to your configuration file.

### Comparing Models

To generate a qualitative side-by-side visual comparison of the anomaly maps produced by all studied models, you can use the `compare_models.py` script. The script generates a single combined overview and individual paper-ready images.

```bash
python -m anomaly_evaluation.compare_models \
  --input '/path/to/dataset/images/*.png' \
  --method msp \
  --erfnet_weights '/path/to/erfnet_pretrained.pth' \
  --eomt_cityscapes_weights '/path/to/eomt_cityscapes.bin' \
  --eomt_cityscapes_config 'eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml' \
  --eomt_coco_weights '/path/to/eomt_coco.bin' \
  --eomt_coco_config 'eomt/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml' \
  --eomt_ft_weights '/path/to/finetuned/checkpoint.ckpt' \
  --eomt_ft_config 'eomt/configs/dinov2/cityscapes/semantic/eomt_finetune_base_640.yaml'
```

🔧 Replace `/path/to/dataset/images/*.png` with the actual path to your dataset images.
🔧 Replace the `/path/to/...` placeholder paths with the actual paths to your trained weights and configuration files.

By default, the script evaluates the `msp` method and looks for the standard checkpoints in the directories. Models without available weights are automatically skipped.

### Temperature Scaling

You can evaluate the effect of temperature scaling on the calibration of the anomaly scores by providing one or more temperature values using the `--temperature` argument. **Note that if multiple temperatures are provided, the script computes the results efficiently by reusing the same network logits, avoiding redundant forward passes:**
  ```bash
  python -m anomaly_evaluation.evalAnomaly \
    --model_type erfnet \
    --input '/path/to/dataset/images/*.png' \
    --weights '/path/to/model_weights.pth' \
    --temperature 0.5 0.75 1.0 1.5 2.0
  ```

  🔧 Replace `/path/to/dataset/images/*.png` and `/path/to/model_weights.pth` with your actual paths.

