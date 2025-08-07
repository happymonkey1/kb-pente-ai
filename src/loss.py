import torch
import torch.nn.functional as F

def policy_loss(logits, target_probs):
    log_probs = F.log_softmax(logits, dim=1)

    return -torch.mean(torch.sum(target_probs * log_probs, dim=1))

def value_loss(pred, target):
    pred = pred.squeeze(-1)
    return F.mse_loss(pred, target)
