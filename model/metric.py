import torch
import numpy as np


def iou_score(output, target, threshold=0.8, eps=1e-8):
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

    intersection = (output * target).sum(1)
    union = torch.max(output, target).sum(1) + eps
    # print((intersection / union).shape)
    return torch.mean(intersection / union)

def mean_iou(output, target, threshold=0.8, eps=1e-8):
    b = output.shape[0]
    output = output.view(b, -1)
    target = target.view(b, -1).float()

    output = (output > threshold).float()
    
    output_empty = torch.max(output, dim=1, keepdim=True)[0] == 0
    target_empty = torch.max(target, dim=1, keepdim=True)[0] == 0
    n_TN = (output_empty * target_empty).sum().float()

    intersection = (output * target).sum(1, keepdim=True)
    union = torch.max(output, target).sum(1) + eps
    score = intersection / union
    # print(score)
    hits = torch.cat([(score > t) for t in np.arange(0.5, 1.0, 0.05)], dim=1).float()
    prec = torch.sum(hits, dim=0) / (b - n_TN + eps)
    return torch.mean(prec)

if __name__ == '__main__':
    out = torch.ones(10, 1, 101, 101) > 0.2
    tar = torch.ones(10, 1, 101, 101) > 0.3
    print(mean_iou(out, tar))