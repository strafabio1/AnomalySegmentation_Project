import torch
import torch.nn.functional as F

def get_msp_score(logits, temperature=1.0):
    logits = logits / temperature
    probs = F.softmax(logits, dim=1) 
    max_prob, _ = torch.max(probs, dim=1)
    return 1.0 - max_prob

def get_max_logit_score(logits, temperature=1.0):
    logits = logits / temperature
    max_logit, _ = torch.max(logits, dim=1)
    return 1.0 - max_logit

def get_max_entropy_score(logits, temperature=1.0):
    logits = logits / temperature
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy

def get_rba_score(pred_logits, pred_masks):
    prob_queries = F.softmax(pred_logits, dim=-1)
    prob_masks = torch.sigmoid(pred_masks)
    
    max_class_prob, _ = prob_queries.max(dim=-1)
    max_class_prob = max_class_prob.unsqueeze(-1).unsqueeze(-1)
    
    query_pixel_scores = max_class_prob * prob_masks
    inlier_score, _ = query_pixel_scores.max(dim=1)
    
    return 1.0 - inlier_score