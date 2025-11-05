import torch
import torch.nn as nn

def make_bce_loss(pos_weight=None):
    if pos_weight is not None:
        pos_w = torch.tensor([float(pos_weight)], dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=pos_w)
    return nn.BCEWithLogitsLoss()
