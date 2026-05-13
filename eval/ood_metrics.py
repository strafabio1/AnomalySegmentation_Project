import numpy as np
from sklearn.metrics import roc_curve, average_precision_score

def fpr_at_95_tpr(val_out, val_label):
    """
    Computes the False Positive Rate (FPR) when the 
    True Positive Rate (TPR) reaches 95%.
    """
    fpr, tpr, thresholds = roc_curve(val_label, val_out)
    idx = np.argmax(tpr >= 0.95)
    return fpr[idx]

def calc_metrics(ood_gts, anomaly_scores):
    """
    Separates pixels into anomalies (1) and normal/in-distribution (0), 
    concatenates the results, and computes AUPRC and FPR95.
    """
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)
    
    return prc_auc, fpr