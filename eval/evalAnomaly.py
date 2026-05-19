import os
import glob
import torch
import numpy as np
import cv2
from PIL import Image
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor
import torch.nn.functional as F

from ood_metrics import calc_metrics
from post_hoc import get_msp_score, get_max_logit_score, get_max_entropy_score, get_rba_score, get_eomt_msp_score, get_eomt_max_logit_score, get_eomt_max_entropy_score
from model_builder import load_erfnet, load_eomt


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
            'MaxLogit': [],
            'MaxEntropy': []
        }
        if args.model_type == 'eomt':
            anomaly_scores_dict[t]['RbA'] = []
    
    ood_gts_list = []

    print("\nStarting evaluation loop...")
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        
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
        
        ood_gts_list.append(ood_gts)
        
        with torch.no_grad():
            if args.model_type == 'erfnet':
                img = Image.open(path).convert('RGB')
                images = erfnet_transform(img).unsqueeze(0).float().to(device)
                
                logits = model(images)
                
                for t in args.temperature:
                    res_maxlogit = get_max_logit_score(logits)[0].cpu().numpy()
                    res_msp = get_msp_score(logits, temperature=t)[0].cpu().numpy()
                    res_maxentropy = get_max_entropy_score(logits)[0].cpu().numpy()
                    
                    anomaly_scores_dict[t]['MaxLogit'].append(res_maxlogit)
                    anomaly_scores_dict[t]['MSP'].append(res_msp)
                    anomaly_scores_dict[t]['MaxEntropy'].append(res_maxentropy)
                
            elif args.model_type == 'eomt':
                img_bgr = cv2.imread(path)
                img_resized = cv2.resize(img_bgr, (eval_size[1], eval_size[0]), interpolation=cv2.INTER_LINEAR)
                images = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)
                
                outputs = model(images)
                p_logits = outputs[1][-1]
                p_masks = outputs[0][-1]
                
                def upscale_score(s_tensor):
                    s_tensor = s_tensor.unsqueeze(0) 
                    s_upscaled = F.interpolate(s_tensor, size=eval_size, mode='bilinear', align_corners=False)
                    return s_upscaled.squeeze().cpu().numpy()

                for t in args.temperature:
                    res_maxlogit = get_eomt_max_logit_score(p_logits, p_masks)
                    res_msp = get_eomt_msp_score(p_logits, p_masks, temperature=t)
                    res_maxentropy = get_eomt_max_entropy_score(p_logits, p_masks)
                    res_rba = get_rba_score(p_logits, p_masks)
                    
                    anomaly_scores_dict[t]['MaxLogit'].append(upscale_score(res_maxlogit))
                    anomaly_scores_dict[t]['MSP'].append(upscale_score(res_msp))
                    anomaly_scores_dict[t]['MaxEntropy'].append(upscale_score(res_maxentropy))
                    anomaly_scores_dict[t]['RbA'].append(upscale_score(res_rba))
                                

        torch.cuda.empty_cache()

    if len(ood_gts_list) == 0:
        print("Error: No valid evaluation images found containing anomalies.")
        return

    ood_gts = np.array(ood_gts_list)
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(str(args.input[0]))))
    
    for t in args.temperature:
        print(f"\n--- {args.model_type.upper()} EVALUATION RESULTS on {dataset_name} (Temperature: {t}) ---")
        for method_name, scores_list in anomaly_scores_dict[t].items():
            prc_auc, fpr = calc_metrics(ood_gts, np.array(scores_list))
            print(f"{method_name:<12} -> AUPRC: {prc_auc*100.0:.2f} | FPR@TPR95: {fpr*100.0:.2f}")

if __name__ == '__main__':
    main()