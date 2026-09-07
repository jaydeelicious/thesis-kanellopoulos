import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class SigmoidFocalLoss(nn.Module):
    ''' ... '''
    def __init__(
        self,
        alpha: float = -1,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        ''' ... '''
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ''' ... '''
        
        valid_mask = (target >= 0)
        input = input[valid_mask]     # [N]
        target = target[valid_mask]   # [N]

        target_bin = (target == 2).float()    # [N], in {0,1}

        ce_loss = F.binary_cross_entropy_with_logits(input, target_bin, reduction='none')
        p = torch.sigmoid(input)
        p_t = p * target_bin + (1 - p) * (1 - target_bin)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target_bin + (1 - self.alpha) * (1 - target_bin)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
