import torch
import torch.nn.functional as F


# METHODS FOR PIXEL-BASED ARCHITECTURES (ERFNet)

def get_msp_score(logits, temperature=1.0):
    logits = logits / temperature
    probs = F.softmax(logits, dim=1) 
    max_prob, _ = torch.max(probs, dim=1)
    return 1.0 - max_prob

def get_max_logit_score(logits, temperature=1.0):
    logits = logits / temperature
    max_logit, _ = torch.max(logits, dim=1)
    return  -max_logit

def get_max_entropy_score(logits, temperature=1.0):
    logits = logits / temperature
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy



# METHODS FOR MASK-BASED ARCHITECTURES (EoMT)

def get_eomt_logits(pred_logits, pred_masks):
    P = pred_logits 
    M = torch.sigmoid(pred_masks)
    L = torch.einsum('bnc,bnhw->bchw', P, M)
    return L

def get_eomt_pixel_probs(pred_logits, pred_masks, temperature=1.0):
    class_probs = F.softmax(pred_logits / temperature, dim=-1)
    mask_probs = torch.sigmoid(pred_masks)
    pixel_probs = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
    pixel_probs_known = pixel_probs[:, :-1, :, :]
    return pixel_probs_known

def get_eomt_msp_score(pred_logits, pred_masks, temperature=1.0):
    probs = get_eomt_pixel_probs(pred_logits, pred_masks, temperature)
    max_prob, _ = torch.max(probs, dim=1)
    return 1.0 - max_prob

def get_eomt_max_logit_score(pred_logits, pred_masks, temperature=1.0):
    L = get_eomt_logits(pred_logits, pred_masks)
    L_known = L[:, :-1, :, :]
    L_known = L_known / temperature
    max_logit, _ = torch.max(L_known, dim=1)
    return -max_logit

def get_eomt_max_entropy_score(pred_logits, pred_masks, temperature=1.0):
    probs = get_eomt_pixel_probs(pred_logits, pred_masks, temperature)
    probs_normalized = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
    log_probs = torch.log(probs_normalized + 1e-8)
    entropy = -torch.sum(probs_normalized * log_probs, dim=1)
    return entropy

def get_rba_score(pred_logits, pred_masks):
    L = get_eomt_logits(pred_logits, pred_masks)
    L_known = L[:, :-1, :, :]
    rba_score = -torch.sum(torch.tanh(L_known), dim=1)
    return rba_score