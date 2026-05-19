<h1 align="center">Comprehensive Road Scene Understanding for Autonomous Driving</h1>

<h3 align="center">FAIMDL Exam Project - A.Y. 2025/2026</h3>

## Project Description
This repository contains the codebase for the **Anomaly Segmentation in Road Scenes** project.
The main objective of this work is the implementation, training, and comparative evaluation of semantic segmentation architectures (specifically the **ERFNet** baseline and the advanced **EoMT** model) for the task of *Anomaly Segmentation* and *Out-Of-Distribution (OOD) Detection* in urban autonomous driving scenarios.

The implemented pipeline supports anomaly mask extraction and the computation of standard evaluation metrics, such as AUPRC and FPR@TPR95, using several post-hoc scoring methods. Pixel-based baselines include MSP, MaxLogit, and MaxEntropy, while mask-based EoMT evaluations additionally support RbA. Temperature scaling is also considered as an additional calibration baseline.

## Authors
* **Simone Duma (s353855)**
* **Giovanni Indiano (s357942)**
* **Paolo Malugani (s359857)**
* **Fabio Stradiotti (s359415)**
---

## Repository Structure

For detailed instructions on running the scripts, please refer to the specific `README.md` files located inside each directory:

* [**anomaly_evaluation**](anomaly_evaluation/): Contains the unified pipeline (`evalAnomaly.py`) for anomaly extraction and evaluation with both models. Here you can also find the scripts for computing metrics and scoring functions.
* [**erfnet**](erfnet/): Contains the ERFNet architecture code, pretrained-weight directory, utilities, and standard semantic segmentation evaluation scripts.
* [**eomt**](eomt/): Contains the EoMT model code adapted for this project, including configuration files, training and fine-tuning scripts for Cityscapes, checkpoint-related resources, and the `eval_mIou.py` script for standard semantic segmentation evaluation. The anomaly segmentation evaluation of EoMT is handled by the unified pipeline in `anomaly_evaluation/`.

## Installation and Requirements

To run the scripts and ensure that all internal Python imports work correctly from any directory, you must install the dependencies and the project package in editable mode. Execute the following commands from the project root:

```bash
pip install -r requirements.txt
pip install -e .
```
