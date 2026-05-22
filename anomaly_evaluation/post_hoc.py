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

def get_eomt_known_class_scores(pred_logits, pred_masks, temperature=1.0):
    class_probs = F.softmax(pred_logits / temperature, dim=-1)
    class_probs_known = class_probs[..., :-1]
    mask_probs = torch.sigmoid(pred_masks)
    known_scores = torch.einsum("bqc,bqhw->bchw", class_probs_known, mask_probs)
    return known_scores

def get_eomt_msp_score(pred_logits, pred_masks, temperature=1.0):
    known_scores = get_eomt_known_class_scores(pred_logits, pred_masks, temperature)
    max_prob, _ = torch.max(known_scores, dim=1)
    return 1.0 - max_prob

def get_eomt_max_logit_score(pred_logits, pred_masks, temperature=1.0):
    known_scores = get_eomt_known_class_scores(pred_logits, pred_masks, temperature)
    max_score, _ = torch.max(known_scores, dim=1)
    return -max_score

def get_eomt_max_entropy_score(pred_logits, pred_masks, temperature=1.0):
    known_scores = get_eomt_known_class_scores(pred_logits, pred_masks, temperature)
    eps = 1e-8
    probs = known_scores / (known_scores.sum(dim=1, keepdim=True) + eps)
    log_probs = torch.log(probs + eps)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy

def get_rba_score(pred_logits, pred_masks, temperature=1.0):
    known_scores = get_eomt_known_class_scores(pred_logits, pred_masks, temperature)
    rba_score = -torch.sum(torch.tanh(known_scores), dim=1)
    return rba_score