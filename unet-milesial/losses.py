import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

def dice_coeff(preds, targets):
    """Dice coeff for batches. Pretty much just calls DiceCoeff.forward() a 
       bunch of times, sums, and averages."""
    if preds.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(preds, targets)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

def iou(preds, targets):
    """Intersection over union for evaluation. Expects batches."""
    smooth = 1e-6
    preds = preds.squeeze(1).int()  # BATCH x 1 x H x W => BATCH x H x W
    targets = targets.squeeze(1).int()
    
    intersection = (preds & targets).float().sum((1, 2))
    union = (preds | targets).float().sum((1, 2))    
    
    iou = (intersection + smooth) / (union + smooth)  # smooth avoids 0/0

    return iou.mean()


class DiceCoeff(Function):
    """Dice coefficient. Can be used as loss since it has a .backward() method
       but BCE has been better for training on the CT data. Method .forward()
       useful as a validation metric."""
    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


class FocalLoss(nn.Module):
    """Focal loss (from Chuong's code)"""
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, 
                                                          targets, 
                                                          reduction = 'none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, 
                                              targets, 
                                              reduction = 'none')
        
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class MixedLoss(nn.Module):
    """Mixed loss (from Chuong's code). Combination of Focal and Dice losses."""
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(4.0, gamma, logits=True, reduce=False)
        
    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target, ) - torch.log(dice_coeff(input, target))
        loss = self.focal(input, target)
        return loss.mean()

