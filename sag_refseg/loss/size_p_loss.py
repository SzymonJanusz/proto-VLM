"""Size penalty / focal loss (ported from kdwonn/SaG)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2norm(x):
    return F.normalize(x, p=2, dim=-1)


class SizeFocalLoss(nn.Module):
    def __init__(self, gamma=5, penalty=0.01, size_average=True):
        super(SizeFocalLoss, self).__init__()
        self.gamma = gamma
        self.penalty = penalty
        self.size_average = size_average
        self.loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, x):
        loss, losses = 0, dict()
        labels = torch.ones_like(x) / x.shape[-1]
        x = (x - x.min(dim=-1, keepdims=True)[0]) / (
            x.max(dim=-1, keepdims=True)[0] - x.min(dim=-1, keepdims=True)[0] + 1e-9)
        loss = self.loss(F.softmax(x.reshape(-1, x.shape[-1]), dim=-1).log(),
                        labels.reshape(-1, x.shape[-1]))
        losses['mask_p_loss'] = loss
        return loss, losses
