# ERFNet Baseline

This directory contains the codebase, evaluation scripts, and utilities for the **ERFNet** (Efficient Residual Factorized Network) architecture. ERFNet serves as the baseline semantic segmentation model for the Anomaly Segmentation Project.

This module has been cleanly separated from the modern `eomt` framework and from the anomaly evaluation suite in order to keep the project modular, easier to maintain, and easier to compare against other models.

## Requirements

Ensure you have installed the project's main dependencies before running the scripts in this module. You can install them from the repository root:

```bash
pip install -r requirements.txt
```

## Directory Structure

- **`model/`**: Contains the PyTorch architecture definitions for ERFNet.
  - `erfnet.py`: Standard ERFNet implementation.
  - `erfnet_nobn.py`: ERFNet implementation without Batch Normalization.

- **`eval/`**: Original ERFNet-specific evaluation scripts used to validate standard semantic segmentation performance, such as mIoU on Cityscapes.

- **`utils/`**: Helper files used by the ERFNet evaluation scripts.
  - `dataset.py`: PyTorch `Dataset` loaders for Cityscapes and VOC12, including label mappings.
  - `transform.py`: Custom PyTorch image transformations, including the logic to colorize inference masks using the Cityscapes color palette.

- **`trained_models/`**: Directory intended to store ERFNet `.pth` checkpoint files and pretrained weights.

## Evaluation

**⚠️ IMPORTANT: All commands must be executed from the root directory of the repository.**

To calculate the mean Intersection over Union (mIoU) and per-class IoU for the ERFNet model on the Cityscapes validation or training set, you can use the `eval_iou.py` script located in the `eval/` directory.

**Options:**
- `--datadir`: Specify the path to your Cityscapes dataset.
- `--subset`: Choose the subset to evaluate (`val` or `train`).
- `--loadWeights`: Filename of the pretrained weights (default is `erfnet_pretrained.pth`).

**Example:**
```bash
python -m erfnet.eval.eval_iou --datadir /path/to/cityscapes/ --subset val
```

## Usage Note

Anomaly segmentation benchmarking on ERFNet, including methods such as MaxLogit, MSP, and MaxEntropy, is **not** performed inside this directory.

To evaluate ERFNet for **Anomaly Segmentation**, use the unified `evalAnomaly.py` script located in the `anomaly_evaluation` folder at the root of the repository.