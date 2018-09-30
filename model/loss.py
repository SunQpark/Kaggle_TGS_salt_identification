import torch
import torch.nn.functional as F
from .lovasz_losses import lovasz_hinge


def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)

def mse_loss(output, target):
    return F.mse_loss(output, target)

def l1_loss(output, target):
    return F.l1_loss(output, target)

def bce_with_l1_loss(output, target, weight_l1=1.0):
    return (bce_loss(output, target) + weight_l1 * l1_loss(output, target)) / (1.0 + weight_l1)

def lovasz_with_bce(output, target):
    return (lovasz_hinge(output, target) + bce_loss(output, target) + l1_loss(output, target)) / 3
