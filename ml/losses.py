import torch
from torch import nn
import torch.nn.functional as F

# https://stackoverflow.com/questions/67230305/i-want-to-confirm-which-of-these-methods-to-calculate-dice-loss-is-correct
class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs, target):
        inputs = torch.sigmoid(inputs)
        num = target.shape[0]
        inputs = inputs.reshape(num, -1)
        target = target.reshape(num, -1)

        intersection = (inputs * target).sum(1)
        union = inputs.sum(1) + target.sum(1)
        dice = (2. * intersection) / (union + 1e-8)
        dice = dice.sum()/num
        return 1 - dice

# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
class DiceLoss2(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1e-8):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection)/(inputs.sum() + targets.sum() + smooth)
        return 1 - dice
    
class WeightedBCE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        weights = (targets != 1) * 1    # Non low confidence will have weight of 1
        return F.binary_cross_entropy_with_logits(
            input=inputs,
            target=targets,
            weight=weights
        )

class WeightedDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1e-8):
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # non_lc_indexes = targets[targets != 1]
        inputs = inputs[targets != 1]
        targets = targets[targets != 1]
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection)/(inputs.sum() + targets.sum() + smooth)
        return 1 - dice
