import sys
from pathlib import Path

import argparse
import yaml
import importlib
import warnings
import numpy as np
import torch
from torch.nn import functional as F
from torch.amp.autocast_mode import autocast
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm
from lightning import seed_everything

IGNORE_INDEX = 255
SINK_CLASS = 19

CLASS_INFO_17 = [
    (0, "road"), (1, "sidewalk"), (2, "building"), (3, "wall"), (4, "fence"),
    (6, "traffic light"), (7, "traffic sign"), (8, "vegetation"), (9, "terrain"),
    (10, "sky"), (11, "person"), (13, "car"), (14, "truck"), (15, "bus"),
    (16, "train"), (17, "motorcycle"), (18, "bicycle")
]

CLASS_INFO_19 = [
    (0, "road"), (1, "sidewalk"), (2, "building"), (3, "wall"), (4, "fence"),
    (5, "pole"), (6, "traffic light"), (7, "traffic sign"), (8, "vegetation"),
    (9, "terrain"), (10, "sky"), (11, "person"), (12, "rider"), (13, "car"),
    (14, "truck"), (15, "bus"), (16, "train"), (17, "motorcycle"), (18, "bicycle")
]

def get_coco_to_cityscapes_mapping(device):
    mapping_tensor = torch.full((256,), SINK_CLASS, dtype=torch.long, device=device)
    mapping_tensor[0] = 11    # person -> person
    mapping_tensor[1] = 18    # bicycle -> bicycle
    mapping_tensor[2] = 13    # car -> car
    mapping_tensor[3] = 17    # motorcycle -> motorcycle

    mapping_tensor[5] = 15    # bus -> bus
    mapping_tensor[6] = 16    # train -> train
    mapping_tensor[7] = 14    # truck -> truck

    mapping_tensor[9] = 6     # traffic light -> traffic light
    mapping_tensor[11] = 7    # stop sign -> traffic sign

    mapping_tensor[58] = 8    # potted plant -> vegetation

    mapping_tensor[86] = 2    # door-stuff -> building
    mapping_tensor[88] = 8    # flower -> vegetation
    mapping_tensor[90] = 9    # gravel -> terrain
    mapping_tensor[91] = 2    # house -> building

    mapping_tensor[100] = 0   # road -> road
    mapping_tensor[101] = 2   # roof -> building
    mapping_tensor[102] = 9   # sand -> terrain

    mapping_tensor[109] = 3   # wall-brick -> wall
    mapping_tensor[110] = 3   # wall-stone -> wall
    mapping_tensor[111] = 3   # wall-tile -> wall
    mapping_tensor[112] = 3   # wall-wood -> wall

    mapping_tensor[114] = 2   # window-blind -> building
    mapping_tensor[115] = 2   # window-other -> building

    mapping_tensor[116] = 8   # tree-merged -> vegetation
    mapping_tensor[117] = 4   # fence-merged -> fence
    mapping_tensor[119] = 10  # sky-other-merged -> sky

    mapping_tensor[123] = 1   # pavement-merged -> sidewalk
    mapping_tensor[125] = 9   # grass-merged -> terrain
    mapping_tensor[126] = 9   # dirt-merged -> terrain

    mapping_tensor[129] = 2   # building-other-merged -> building
    mapping_tensor[130] = 9   # rock-merged -> terrain
    mapping_tensor[131] = 3   # wall-other-merged -> wall
    
    return mapping_tensor

def build_and_load_model(config, data_meta, weights_path, device):
    warnings.filterwarnings(
        "ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
    )
    # Load encoder
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
    encoder = encoder_cls(img_size=data_meta.img_size, **encoder_cfg.get("init_args", {}))

    # Load network
    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}

    network = network_cls(
        masked_attn_enabled=False,
        num_classes=data_meta.num_classes,
        encoder=encoder,
        **network_kwargs,
    )

    # Load Lightning module
    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}

    if "stuff_classes" in config["data"].get("init_args", {}):
        model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

    name = config.get("trainer", {}).get("logger", {}).get("init_args", {}).get("name", "")
    is_dinov3 = "dinov3" in name

    if is_dinov3:
        model_kwargs["ckpt_path"] = weights_path
        model_kwargs["delta_weights"] = True

    model = (
        lit_cls(
            img_size=data_meta.img_size,
            num_classes=data_meta.num_classes,
            network=network,
            **model_kwargs,
        )
        .eval()
        .to(device)
    )

    if not is_dinov3:
        try:
            ckpt = torch.load(weights_path, map_location=device)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
            model.load_state_dict(state_dict, strict=False)
            print(f"Weights successfully loaded from {weights_path}!")
        except Exception as e:
            print(f"Error loading weights from {weights_path}: {e}")

    return model

def infer_semantic(model_sem, data_val, img, target, device):
    device_type = "cuda" if device.type == "cuda" else "cpu"
    with torch.no_grad(), autocast(device_type=device_type, dtype=torch.float16 if device_type == "cuda" else torch.float32):
        imgs = [img.to(device)]
        img_sizes = [img.shape[-2:] for img in imgs]

        crops, origins = model_sem.window_imgs_semantic(imgs)
        mask_logits_per_layer, class_logits_per_layer = model_sem(crops)

        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], data_val.img_size, mode="bilinear"
        )

        crop_logits = model_sem.to_per_pixel_logits_semantic(
            mask_logits, class_logits_per_layer[-1]
        )
        logits = model_sem.revert_window_logits_semantic(crop_logits, origins, img_sizes)
        preds = logits[0].argmax(0).cpu()

    pred_array = preds.numpy()
    target_array = model_sem.to_per_pixel_targets_semantic([target], IGNORE_INDEX)[0].numpy()
    return pred_array, target_array

def main():
    parser = argparse.ArgumentParser(description="Evaluate Semantic/Panoptic models (mIoU).")
    parser.add_argument("--config", type=str, required=True, help="Path to the model YAML config.")
    parser.add_argument("--weights", type=str, required=True, help="Path to the model weights.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--eval_type", type=str, choices=["mapped_17", "all_19"], required=True, help="Type of evaluation to perform.")
    args = parser.parse_args()

    seed_everything(0, verbose=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load Config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Initialize Dataset
    data_module_name, class_name = config["data"]["class_path"].rsplit(".", 1)
    data_module = getattr(importlib.import_module(data_module_name), class_name)
    data_kwargs = config["data"].get("init_args", {})
    
    dataset = data_module(
        path=args.data_path,
        batch_size=1,
        num_workers=0,
        check_empty_targets=False,
        **data_kwargs
    )
    dataset.setup()
    val_dataloader = dataset.val_dataloader()

    # Load Model
    print("Initializing model...")
    model = build_and_load_model(config, dataset, args.weights, device)

    if args.eval_type == "all_19" and model.num_classes != 19:
        import sys
        print(f"Error: Requested 'all_19' evaluation but the model produces {model.num_classes} classes instead of 19.")
        sys.exit(1)

    is_coco = "eomt_coco.bin" in args.weights

    # Setup Metrics and Mapping
    if is_coco:
        metric = MulticlassJaccardIndex(num_classes=20, ignore_index=IGNORE_INDEX, average="none").to(device)
        mapping_tensor = get_coco_to_cityscapes_mapping(device)
    else:
        metric = MulticlassJaccardIndex(num_classes=19, ignore_index=IGNORE_INDEX, average="none").to(device)

    if args.eval_type == "mapped_17":
        class_info = CLASS_INFO_17
    else:
        class_info = CLASS_INFO_19
        
    common_class_ids_tensor = torch.tensor([cid for cid, _ in class_info], dtype=torch.long, device=device)

    print(f"Starting evaluation (eval_type={args.eval_type}, mapping_active={is_coco})...")
    
    with torch.inference_mode():
        for batch in tqdm(val_dataloader, desc="Processing"):
            batched_imgs, batched_targets = batch
            img = batched_imgs[0]
            target = batched_targets[0]

            preds_np, target_array_np = infer_semantic(model, dataset, img, target, device)
            preds_tensor = torch.as_tensor(preds_np, dtype=torch.long, device=device)
            target_tensor = torch.as_tensor(target_array_np, dtype=torch.long, device=device)

            if args.eval_type == "mapped_17":
                # Mask out non-common classes (e.g. pole and rider) in targets
                target_common_tensor = target_tensor.clone()
                target_common_tensor[~torch.isin(target_common_tensor, common_class_ids_tensor)] = IGNORE_INDEX
            else:
                target_common_tensor = target_tensor
                
            if is_coco:
                # Map COCO predictions to Cityscapes
                preds_mapped_tensor = mapping_tensor[preds_tensor]
                metric.update(preds_mapped_tensor, target_common_tensor)
            else:
                metric.update(preds_tensor, target_common_tensor)

    iou_array = metric.compute()

    print("\n" + "=" * 55)
    print(f"{'CLASS':<20} | {'IoU (%)':<22}")
    print("-" * 55)

    valid_values = []
    for cid, name in class_info:
        iou = iou_array[cid]
        if torch.isnan(iou):
            iou_str = f"{'N/A':>18}"
        else:
            iou_str = f"{iou.item() * 100:>18.2f}"
            valid_values.append(iou)
        print(f"{name:<20} | {iou_str}")

    valid_tensor = torch.stack(valid_values) if valid_values else None
    miou = valid_tensor.mean().item() * 100 if valid_tensor is not None else 0.0

    print("-" * 55)
    print(f"{'TOTAL mIoU':<20} | {miou:>18.2f}")
    print("=" * 55)

if __name__ == "__main__":
    main()
