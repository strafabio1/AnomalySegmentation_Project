import sys
from pathlib import Path

import os
import glob
import torch
import numpy as np
import cv2
from PIL import Image
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor
import torch.nn.functional as F

from anomaly_evaluation.ood_metrics import calc_metrics
from anomaly_evaluation.post_hoc import get_msp_score, get_max_logit_score, get_max_entropy_score, get_rba_score, get_eomt_msp_score, get_eomt_max_logit_score, get_eomt_max_entropy_score
from anomaly_evaluation.model_builder import load_erfnet, load_eomt


def load_and_prepare_mask(pathGT, target_transform):
    """
    Loads the ground truth mask and maps dataset-specific labels 
    to a unified anomaly format: 1 for anomaly, 0 for normal/in-distribution, 
    and 255 for ignored regions.
    """
    if "RoadObsticle21" in pathGT:
        pathGT = pathGT.replace("webp", "png")
    if "fs_static" in pathGT:
        pathGT = pathGT.replace("jpg", "png")                
    if "RoadAnomaly" in pathGT:
        pathGT = pathGT.replace("jpg", "png")  
    
    mask = Image.open(pathGT)
    mask = target_transform(mask)
    ood_gts = np.array(mask)

    if "RoadAnomaly" in pathGT:
        ood_gts = np.where((ood_gts==2), 1, ood_gts)
    if "LostAndFound" in pathGT:
        ood_gts = np.where((ood_gts==0), 255, ood_gts)
        ood_gts = np.where((ood_gts==1), 0, ood_gts)
        ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)
    if "Streethazard" in pathGT:
        ood_gts = np.where((ood_gts==14), 255, ood_gts)
        ood_gts = np.where((ood_gts<20), 0, ood_gts)
        ood_gts = np.where((ood_gts==255), 1, ood_gts)
        
    return ood_gts


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+")
    parser.add_argument('--model_type', required=True, choices=['erfnet', 'eomt'])
    parser.add_argument('--weights', required=True)
    parser.add_argument('--config', default="")
    parser.add_argument('--temperature', type=float, nargs='+', default=[1.0])
    args = parser.parse_args()
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nInitializing {args.model_type.upper()} model...")
    if args.model_type == 'erfnet':
        model = load_erfnet(args.weights).to(device)
    elif args.model_type == 'eomt':
        assert args.config, "EoMT requires a --config (.yaml) file"
        model = load_eomt(args.config, args.weights, device).to(device)
        
    model.eval()

    if args.model_type == 'eomt':
        if 'cityscapes' in args.weights.lower():
            eval_size = (1024, 1024)
        else:
            eval_size = (640, 640)
    else:
        eval_size = (512, 1024)

    erfnet_transform = Compose([
        Resize(eval_size, Image.BILINEAR), 
        ToTensor()    
        ])
    target_transform = Compose([Resize(eval_size, Image.NEAREST)])

    anomaly_scores_dict = {}
    for t in args.temperature:
        anomaly_scores_dict[t] = {
            'MSP': [],
            'MaxEntropy': []
        }

    invariant_scores_dict = {
        'MaxLogit': []
    }
    if args.model_type == 'eomt':
        invariant_scores_dict['RbA'] = []
    
    ood_gts_list = []

    print("\nStarting evaluation loop...")
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        
        pathGT = path.replace("images", "labels_masks")                
        
        ood_gts = load_and_prepare_mask(pathGT, target_transform)
            
        
        if 1 not in np.unique(ood_gts):
            continue
        
        ood_gts_list.append(ood_gts)
        
        with torch.no_grad():
            if args.model_type == 'erfnet':
                img = Image.open(path).convert('RGB')
                images = erfnet_transform(img).unsqueeze(0).to(device)
                logits = model(images)
                
                res_maxlogit = get_max_logit_score(logits)[0].cpu().numpy()
                invariant_scores_dict['MaxLogit'].append(res_maxlogit)
                
                for t in args.temperature:
                    res_msp = get_msp_score(logits, temperature=t)[0].cpu().numpy()
                    res_maxentropy = get_max_entropy_score(logits, temperature=t)[0].cpu().numpy()
                    anomaly_scores_dict[t]['MSP'].append(res_msp)
                    anomaly_scores_dict[t]['MaxEntropy'].append(res_maxentropy)
                
            elif args.model_type == 'eomt':
                img = Image.open(path).convert('RGB')
                img_resized = img.resize((eval_size[1], eval_size[0]), Image.BILINEAR)

                img_array = np.array(img_resized)
                images = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float().to(device)
                
                outputs = model(images)
                p_logits = outputs[1][-1]
                p_masks = outputs[0][-1]
                
                def upscale_score(s_tensor):
                    s_tensor = s_tensor.unsqueeze(0) 
                    s_upscaled = F.interpolate(s_tensor, size=eval_size, mode='bilinear', align_corners=False)
                    return s_upscaled.squeeze().cpu().numpy()

                res_maxlogit = get_eomt_max_logit_score(p_logits, p_masks)
                res_rba = get_rba_score(p_logits, p_masks)
                
                invariant_scores_dict['MaxLogit'].append(upscale_score(res_maxlogit))
                invariant_scores_dict['RbA'].append(upscale_score(res_rba))

                for t in args.temperature:
                    res_msp = get_eomt_msp_score(p_logits, p_masks, temperature=t)
                    res_maxentropy = get_eomt_max_entropy_score(p_logits, p_masks, temperature=t)
                    anomaly_scores_dict[t]['MSP'].append(upscale_score(res_msp))
                    anomaly_scores_dict[t]['MaxEntropy'].append(upscale_score(res_maxentropy))
                                
        torch.cuda.empty_cache()

    if len(ood_gts_list) == 0:
        print("Error: No valid evaluation images found containing anomalies.")
        return

    ood_gts = np.array(ood_gts_list)
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(str(args.input[0]))))
    
    invariant_results = {}
    for method_name, scores_list in invariant_scores_dict.items():
        invariant_results[method_name] = calc_metrics(ood_gts, np.array(scores_list))

    all_t_results = {}
    for t in args.temperature:
        t_results = {}
        for method_name, scores_list in anomaly_scores_dict[t].items():
            t_results[method_name] = calc_metrics(ood_gts, np.array(scores_list))
        all_t_results[t] = t_results

    print(f"\n--- {args.model_type.upper()} EVALUATION RESULTS on {dataset_name} ---")
    
    header = f"{'Method':<12} "
    for t in args.temperature:
        header += f"| {f'T={t} AUPRC':^11} | {f'T={t} FPR95':^11} "
    print(header)
    print("-" * len(header))
    
    print_order = ['MSP', 'MaxLogit', 'MaxEntropy']
    if args.model_type == 'eomt':
        print_order.append('RbA')
        
    for method_name in print_order:
        row_str = f"{method_name:<12} "
        for t in args.temperature:
            if method_name in all_t_results[t]:
                prc_auc, fpr = all_t_results[t][method_name]
            elif method_name in invariant_results:
                prc_auc, fpr = invariant_results[method_name]
            else:
                row_str += f"| {'-':^11} | {'-':^11} "
                continue
            row_str += f"| {prc_auc*100.0:11.2f} | {fpr*100.0:11.2f} "
        print(row_str)

if __name__ == '__main__':
    main()