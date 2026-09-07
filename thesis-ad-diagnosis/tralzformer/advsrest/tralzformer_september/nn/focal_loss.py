import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class SoftmaxFocalLoss(nn.Module):
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
        """
        Focal loss for multiclass classification at a single time step.

        input: [B, C] logits
        target: [B] or [B, 1] class indices (0...C-1), or -1 for padding
        """
        if target.ndim == 2:
            target = target.squeeze(1)  # Convert [B, 1] to [B]

        target = target.long()
        B, C = input.shape

        # Filter valid targets (ignore -1 or other padding values)
        valid_mask = (target >= 0) & (target < C)
        input = input[valid_mask]     # [N, C]
        target = target[valid_mask]   # [N]

        if input.numel() == 0:
            return torch.tensor(0.0, device=input.device, requires_grad=True)

        logp = F.log_softmax(input, dim=-1)  # [N, C]
        p = torch.exp(logp)                  # [N, C]

        logp_t = logp[torch.arange(len(target)), target]  # [N]
        p_t = p[torch.arange(len(target)), target]        # [N]

        loss = -((1 - p_t) ** self.gamma) * logp_t

        if self.alpha >= 0:
            loss = self.alpha * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # [N]