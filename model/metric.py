import torch
import numpy as np


def iou_score(output, target, threshold=0.5, eps=1e-8):
    """ 
    param 
        output: binary mask containing model output
        target: binary mask containing ground truth mask
    """
    b, c, w, h = output.shape
    output = (output > threshold).float()
    if torch.max(output) == 0:
        return iou_score(1 - output, 1 - target)
    output = output.view(b, c*w*h)
    target = target.view(b, c*w*h)

    intersection = (output * target).sum(1) + eps
    union = torch.max(output, target).sum(1) + eps
    return torch.mean(intersection / union)


