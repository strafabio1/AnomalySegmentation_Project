import os
import glob
import torch
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor
import torch.nn.functional as F

from ood_metrics import calc_metrics
from post_hoc import get_msp_score, get_max_logit_score, get_max_entropy_score, get_rba_score, get_eomt_msp_score, get_eomt_max_logit_score, get_eomt_max_entropy_score
from model_builder import load_erfnet, load_eomt


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", default="path/to/images/*.webp", nargs="+")
    parser.add_argument('--model_type', required=True, choices=['erfnet', 'eomt'])
    parser.add_argument('--weights', required=True)
    parser.add_argument('--config', default="")
    parser.add_argument('--temperature', type=float, default=1.0)
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

    input_transform = Compose([Resize(eval_size, Image.BILINEAR), ToTensor()])
    target_transform = Compose([Resize(eval_size, Image.NEAREST)])

    anomaly_scores_dict = {
        'MSP': [],
        'MaxLogit': [],
        'MaxEntropy': []
        }
    if args.model_type == 'eomt':
        anomaly_scores_dict['RbA'] = []
    
    ood_gts_list = []

    print("\nStarting evaluation loop...")
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        img = Image.open(path).convert('RGB')
        images = input_transform(img).unsqueeze(0).float().to(device)

        with torch.no_grad():
            if args.model_type == 'erfnet':
                logits = model(images)
                res_maxlogit = get_max_logit_score(logits, temperature=args.temperature)[0].cpu().numpy()
                res_msp = get_msp_score(logits, temperature=args.temperature)[0].cpu().numpy()
                res_maxentropy = get_max_entropy_score(logits, temperature=args.temperature)[0].cpu().numpy()
                
            elif args.model_type == 'eomt':
                outputs = model(images)
                p_logits = outputs[1][-1]
                p_masks = outputs[0][-1]
                
                res_maxlogit = get_eomt_max_logit_score(p_logits, p_masks, temperature=args.temperature)
                res_msp = get_eomt_msp_score(p_logits, p_masks, temperature=args.temperature)
                res_maxentropy = get_eomt_max_entropy_score(p_logits, p_masks, temperature=args.temperature)
                res_rba = get_rba_score(p_logits, p_masks)
                
                def upscale_score(s_tensor):
                    s_tensor = s_tensor.unsqueeze(0) 
                    s_upscaled = F.interpolate(s_tensor, size=eval_size, mode='bilinear', align_corners=False)
                    return s_upscaled.squeeze().cpu().numpy()

                res_maxlogit = upscale_score(res_maxlogit)
                res_msp = upscale_score(res_msp)
                res_maxentropy = upscale_score(res_maxentropy)
                res_rba = upscale_score(res_rba)
                                

        pathGT = path.replace("images", "labels_masks")                
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

        if 1 not in np.unique(ood_gts):
            continue              
        else:
            ood_gts_list.append(ood_gts)
            anomaly_scores_dict['MaxLogit'].append(res_maxlogit)
            anomaly_scores_dict['MSP'].append(res_msp)
            anomaly_scores_dict['MaxEntropy'].append(res_maxentropy)
            if args.model_type == 'eomt':
                anomaly_scores_dict['RbA'].append(res_rba)
            
        torch.cuda.empty_cache()

    if len(ood_gts_list) == 0:
        print("Error: No valid evaluation images found containing anomalies.")
        return

    ood_gts = np.array(ood_gts_list)
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(str(args.input[0]))))
    print(f"--- {args.model_type.upper()} EVALUATION RESULTS on {dataset_name} (Temperature: {args.temperature}) ---")
    for method_name, scores_list in anomaly_scores_dict.items():
        prc_auc, fpr = calc_metrics(ood_gts, np.array(scores_list))
        print(f"{method_name:<12} -> AUPRC: {prc_auc*100.0:.2f} | FPR@TPR95: {fpr*100.0:.2f}")

if __name__ == '__main__':
    main()