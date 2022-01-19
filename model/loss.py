import torch.nn.functional as F
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)

def MSE_loss(output, target):
    return F.mse_loss(output, target)
    
def my_loss(output, target):
    loss = (output - target)**2
    return torch.sqrt(loss.sum())