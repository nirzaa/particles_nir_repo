import torch.nn.functional as F
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)

def MSE_loss(output, target):
    return F.mse_loss(output, target)
    
def my_loss(output, target):
<<<<<<< Updated upstream
    loss = (output - target)**2
    return torch.sqrt(loss.sum())
=======
    loss = (output - target)**2 / target**2
    return loss.mean()
>>>>>>> Stashed changes
