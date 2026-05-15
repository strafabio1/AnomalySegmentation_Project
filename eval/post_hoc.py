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
    """
    Calculates the spatial logits L(x):
    L(x) = sum_n P_n(x) * M_n(x)
    """
    P = pred_logits 
    M = torch.sigmoid(pred_masks)
    L = torch.einsum('bnc,bnhw->bchw', P, M)
    return L

def get_eomt_msp_score(pred_logits, pred_masks, temperature=1.0):
    L = get_eomt_logits(pred_logits, pred_masks)
    #Only known classes
    L_known = L[:, :-1, :, :]
    L_known = L_known / temperature
    probs = F.softmax(L_known, dim=1)
    max_prob, _ = torch.max(probs, dim=1)
    return 1.0 - max_prob

def get_eomt_max_logit_score(pred_logits, pred_masks, temperature=1.0):
    L = get_eomt_logits(pred_logits, pred_masks)
    #Only known classes
    L_known = L[:, :-1, :, :]
    L_known = L_known / temperature
    max_logit, _ = torch.max(L_known, dim=1)
    return -max_logit

def get_eomt_max_entropy_score(pred_logits, pred_masks, temperature=1.0):
    L = get_eomt_logits(pred_logits, pred_masks)
    #Only known classes
    L_known = L[:, :-1, :, :]
    L_known = L_known / temperature
    probs = F.softmax(L_known, dim=1)
    log_probs = F.log_softmax(L_known, dim=1)
    return -torch.sum(probs * log_probs, dim=1)

def get_rba_score(pred_logits, pred_masks):
    L = get_eomt_logits(pred_logits, pred_masks)
    #Only known classes
    L_known = L[:, :-1, :, :]
    rba_score = -torch.sum(torch.tanh(L_known), dim=1)
    
    return rba_score