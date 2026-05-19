# EoMT

This repository contains the codebase for the Anomaly Segmentation Project, which builds upon the foundational architecture of the [EoMT repository](https://github.com/tue-mps/eomt). 

It provides extended scripts and modules to efficiently fine-tune, train, and evaluate EoMT models on the Cityscapes dataset, alongside specialized tools for rigorous Anomaly Detection benchmarking.

You can download our custom pre-trained models on Cityscapes at this [link](https://drive.google.com/drive/folders/1q2vHUzora2nP52fP50zmoQAykWuwoGav?usp=drive_link).

## Requirements Installation

If you don't have Conda installed, install Miniconda and restart your shell:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Then create the environment, activate it, and install the dependencies:

```bash
conda create -n eomt python==3.13.2
conda activate eomt
python3 -m pip install -r requirements.txt
```

[Weights & Biases](https://wandb.ai/) (wandb) is used for experiment logging and visualization. To enable wandb, log in to your account:

```bash
wandb login
```

## Data preparation for training

You do **not** need to unzip any of the downloaded files.  
Simply place them in a directory of your choice and provide that path via the `--data.path` argument.  
The code will read the `.zip` files directly.

**Cityscapes**
```bash
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=<your_username>&password=<your_password>&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
```

🔧 Replace `<your_username>` and `<your_password>` with your actual [Cityscapes](https://www.cityscapes-dataset.com/) login credentials.  

## Usage

### Training & Finetuning

To train EoMT from scratch (not recommended in Colab due to resource constraints):
```bash
python3 main.py fit \
  -c configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset
```
✅ Make sure the total batch size is `devices × batch_size × accumulate_grad_batches = 16`.

**Finetuning on Cityscapes**
To fine-tune a pre-trained COCO EoMT model on Cityscapes, you can control how many blocks of the backbone to unfreeze.

*1. Finetune ONLY the Classification Head (0 blocks unfrozen):*
```bash
python main.py fit \
  -c configs/dinov2/cityscapes/semantic/eomt_finetune_base_640.yaml \
  --data.batch_size 8 \
  --trainer.accumulate_grad_batches 2 \
  --data.path /path/to/Cityscapes \
  --trainer.devices 1 \
  --trainer.logger.init_args.name "finetune_only_Head" \
  --model.network.init_args.unfreeze_last_n_blocks 0 \
  --model.load_ckpt_class_head False \
  --model.ckpt_path /path/to/pytorch_model.bin
```

*2. Finetune unfreezing the last N blocks:*
You can progressively unfreeze more blocks by changing `--model.network.init_args.unfreeze_last_n_blocks` to `1`, `2`, or `3`. For example, to unfreeze the last 3 blocks:
```bash
python main.py fit \
  -c configs/dinov2/cityscapes/semantic/eomt_finetune_base_640.yaml \
  --data.batch_size 8 \
  --trainer.accumulate_grad_batches 2 \
  --data.path /path/to/Cityscapes \
  --trainer.devices 1 \
  --trainer.logger.init_args.name "finetune_unfreeze3" \
  --model.network.init_args.unfreeze_last_n_blocks 3 \
  --model.load_ckpt_class_head False \
  --model.ckpt_path /path/to/pytorch_model.bin
```

🔧 Replace `/path/to/pytorch_model.bin` with the path to the checkpoint to fine-tune.

### Evaluating

**Evaluating mIoU on Cityscapes**
You can use the custom `eval_mIou.py` script to evaluate the Mean Intersection over Union (mIoU) on the Cityscapes validation set. The script intelligently handles mapping if you pass the COCO pre-trained model.

```bash
python eval_mIou.py \
  --config configs/dinov2/cityscapes/semantic/eomt_finetune_base_640.yaml \
  --weights /path/to/pytorch_model.bin \
  --data_path /path/to/Cityscapes \
  --eval_type mapped_17
```

🔧 Replace `/path/to/pytorch_model.bin` with the path to the checkpoint to evaluate.

*Arguments:*
- `--eval_type mapped_17`: Evaluates the model strictly on 17 common classes, automatically ignoring `pole` and `rider` from the targets. If the weights file is `eomt_coco.bin`, it automatically maps the COCO output to the Cityscapes format.
- `--eval_type all_19`: Evaluates the model on all 19 standard Cityscapes classes. **Note:** This will throw an error if the model does not output exactly 19 classes (e.g. if you try to evaluate the raw COCO model with this parameter).
