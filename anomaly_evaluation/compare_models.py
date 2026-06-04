import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import sys

from anomaly_evaluation.model_builder import load_erfnet, load_eomt
from anomaly_evaluation.post_hoc import (
    get_msp_score, get_max_logit_score, get_max_entropy_score,
    get_eomt_msp_score, get_eomt_max_logit_score, get_eomt_max_entropy_score,
)


def build_model_specs(args):
    specs = [
        {
            "name": "ERFNet",
            "type": "erfnet",
            "weights": args.erfnet_weights,
            "config": None,
        },
        {
            "name": "EoMT (Cityscapes)",
            "type": "eomt",
            "weights": args.eomt_cityscapes_weights,
            "config": args.eomt_cityscapes_config,
        },
        {
            "name": "EoMT (COCO)",
            "type": "eomt",
            "weights": args.eomt_coco_weights,
            "config": args.eomt_coco_config,
        },
        {
            "name": "EoMT fine-tuned",
            "type": "eomt",
            "weights": args.eomt_ft_weights,
            "config": args.eomt_ft_config,
        },
    ]

    available = []
    for s in specs:
        if not os.path.isfile(s["weights"]):
            print(f"[warn] weights not found for {s['name']}: {s['weights']} -> skipping")
            continue
        if s["type"] == "eomt" and not (s["config"] and os.path.isfile(s["config"])):
            print(f"[warn] config not found for {s['name']}: {s['config']} -> skipping")
            continue
        available.append(s)
    return available


def eomt_eval_size(weights_path):
    return (1024, 1024) if "cityscapes" in weights_path.lower() else (640, 640)


def anomaly_map_erfnet(model, image_path, method, temperature, device):
    eval_size = (512, 1024)
    transform = Compose([Resize(eval_size, Image.BILINEAR), ToTensor()])
    img = Image.open(image_path).convert("RGB")
    images = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(images)
        if method == "maxlogit":
            score = get_max_logit_score(logits)
        elif method == "msp":
            score = get_msp_score(logits, temperature=temperature)
        elif method == "maxentropy":
            score = get_max_entropy_score(logits, temperature=temperature)
        else:
            raise ValueError(method)
    return score[0].detach().cpu().numpy()


def anomaly_map_eomt(model, image_path, method, temperature, eval_size, device):
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((eval_size[1], eval_size[0]), Image.BILINEAR)
    img_array = np.array(img_resized)
    images = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        outputs = model(images)
        p_logits = outputs[1][-1]
        p_masks = outputs[0][-1]

        if method == "maxlogit":
            score = get_eomt_max_logit_score(p_logits, p_masks)
        elif method == "msp":
            score = get_eomt_msp_score(p_logits, p_masks, temperature=temperature)
        elif method == "maxentropy":
            score = get_eomt_max_entropy_score(p_logits, p_masks, temperature=temperature)
        else:
            raise ValueError(method)

        score = score.unsqueeze(0)
        score = F.interpolate(score, size=eval_size, mode="bilinear", align_corners=False)
    return score.squeeze().detach().cpu().numpy()


def compute_map(spec, model, image_path, method, temperature, device):
    if spec["type"] == "erfnet":
        return anomaly_map_erfnet(model, image_path, method, temperature, device)
    return anomaly_map_eomt(model, image_path, method, temperature,
                            eomt_eval_size(spec["weights"]), device)


def normalize01(m):
    m = m.astype(np.float32)
    lo, hi = np.nanmin(m), np.nanmax(m)
    if hi - lo < 1e-12:
        return np.zeros_like(m)
    return (m - lo) / (hi - lo)


def _draw_row(fig, axes_row, img_path, maps, model_names, cmap, overlay, alpha, show_titles):
    rgb = np.array(Image.open(img_path).convert("RGB"))
    H, W = rgb.shape[:2]

    axes_row[0].imshow(rgb)
    axes_row[0].axis("off")
    if show_titles:
        axes_row[0].set_title("Input", fontsize=11)

    for c, name in enumerate(model_names):
        ax = axes_row[c + 1]
        amap = maps.get(name)
        if show_titles:
            ax.set_title(name, fontsize=11)
        if amap is None:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center")
            ax.axis("off")
            continue
        amap_img = np.array(
            Image.fromarray((normalize01(amap) * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
        ) / 255.0
        if overlay:
            ax.imshow(rgb)
            ax.imshow(amap_img, cmap=cmap, alpha=alpha, vmin=0.0, vmax=1.0)
        else:
            ax.imshow(amap_img, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.axis("off")

    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=list(axes_row), fraction=0.020, pad=0.012)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["low", "high"])
    cbar.set_label("Anomaly score", fontsize=9)


def render_rows(image_paths, maps_per_image, model_names, method, out_path,
                cmap="inferno", overlay=False, alpha=0.5, suptitle=None, dpi=200):
    n_rows = len(image_paths)
    n_cols = 1 + len(model_names)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.2 * n_cols + 0.8, 3.0 * n_rows),
                             squeeze=False, constrained_layout=True)
    for r, (img_path, maps) in enumerate(zip(image_paths, maps_per_image)):
        _draw_row(fig, axes[r], img_path, maps, model_names,
                  cmap, overlay, alpha, show_titles=(r == 0))
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[ok] saved figure -> {out_path}")


def dataset_name_from_input(pattern):
    p = str(pattern).rstrip("/")
    name = os.path.basename(os.path.dirname(os.path.dirname(p)))
    if not name or name in (".", "..") or any(ch in name for ch in "*?[]"):
        return ""
    return name


def build_figure(image_paths, maps_per_image, model_names, method, out_dir, base_name,
                 dataset="", cmap="inferno", overlay=False, alpha=0.5, dpi=200):
    os.makedirs(out_dir, exist_ok=True)

    title = f"Anomaly maps - post-hoc method: {method}" + (f" - {dataset}" if dataset else "")
    summary_path = os.path.join(out_dir, base_name + ".png")
    render_rows(image_paths, maps_per_image, model_names, method, summary_path,
                cmap=cmap, overlay=overlay, alpha=alpha, suptitle=title, dpi=dpi)

    for i, img_path in enumerate(image_paths):
        base = os.path.splitext(os.path.basename(img_path))[0]
        row_out = os.path.join(out_dir, f"row{i:02d}_{base}_{method}.png")
        render_rows([img_path], [maps_per_image[i]], model_names, method, row_out,
                    cmap=cmap, overlay=overlay, alpha=alpha, suptitle=None, dpi=max(dpi, 300))
    print(f"[ok] all outputs in: {out_dir}/  (summary: {os.path.basename(summary_path)}, "
          f"{len(image_paths)} per-row PNG)")



def main():
    parser = ArgumentParser(description="Compare anomaly maps of ERFNet, EoMT and EoMT fine-tuned with the same post-hoc method.")
    parser.add_argument("--input", required=True, nargs="+",
                        help="Image path(s) or a glob, e.g. 'dataset/images/*.png'.")
    parser.add_argument("--method", default="msp",
                        choices=["maxlogit", "msp", "maxentropy"],
                        help="Post-hoc scoring rule applied to every model (default: msp).")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for MSP / MaxEntropy (ignored by MaxLogit).")

    parser.add_argument("--erfnet_weights", default="erfnet/trained_models/erfnet_pretrained.pth")
    parser.add_argument("--eomt_cityscapes_weights", default="eomt/weights/eomt_cityscapes.bin")
    parser.add_argument("--eomt_cityscapes_config", default="eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml")
    parser.add_argument("--eomt_coco_weights", default="eomt/weights/eomt_coco.bin")
    parser.add_argument("--eomt_coco_config", default="eomt/configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml")
    parser.add_argument("--eomt_ft_weights", default="eomt/eomt/finetune_unfreeze3_batch16_final/checkpoints/best.ckpt")
    parser.add_argument("--eomt_ft_config", default="eomt/configs/dinov2/cityscapes/semantic/eomt_finetune_base_640.yaml")

    parser.add_argument("--out_dir", default=None,
                        help="Folder that holds the summary image AND the per-image PNGs "
                             "(default: comparison_<method>_<dataset>/).")
    parser.add_argument("--dataset", default=None,
                        help="Dataset name appended to the folder and summary file "
                             "(default: inferred from the input path, e.g. RoadAnomaly21).")
    parser.add_argument("--overlay", action="store_true", help="Blend the heatmap over the input image.")
    parser.add_argument("--cmap", default="inferno", help="Single colormap used for every anomaly map.")
    parser.add_argument("--max_images", type=int, default=6, help="Cap on number of images to render.")
    parser.add_argument("--dpi", type=int, default=200,
                        help="DPI of the combined figure (per-row PNGs use at least 300).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset if args.dataset is not None else dataset_name_from_input(args.input[0])
    base_name = f"comparison_{args.method}" + (f"_{dataset}" if dataset else "")
    out_dir = args.out_dir or base_name

    image_paths = [p for p in args.input if os.path.isfile(p)][: args.max_images]
    if not image_paths:
        print("Error: no valid input images found.")
        sys.exit(1)

    specs = build_model_specs(args)
    if not specs:
        print("Error: no model could be loaded (check weights/config paths).")
        sys.exit(1)
    model_names = [s["name"] for s in specs]

    maps_per_image = [dict() for _ in image_paths]
    for spec in specs:
        print(f"\nLoading {spec['name']} ({spec['type']}) from {spec['weights']} ...")
        if spec["type"] == "erfnet":
            model = load_erfnet(spec["weights"]).to(device)
        else:
            model = load_eomt(spec["config"], spec["weights"], device).to(device)
        model.eval()

        for i, img_path in enumerate(image_paths):
            try:
                amap = compute_map(spec, model, img_path, args.method, args.temperature, device)
                maps_per_image[i][spec["name"]] = amap
            except Exception as e:
                print(f"[warn] {spec['name']} failed on {img_path}: {e}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    build_figure(image_paths, maps_per_image, model_names, args.method,
                 out_dir, base_name, dataset=dataset,
                 cmap=args.cmap, overlay=args.overlay, dpi=args.dpi)


if __name__ == "__main__":
    main()