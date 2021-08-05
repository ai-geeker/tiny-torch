import tinytorch as torch
from tinytorch.nn import Module
import tinytorch.nn.functional as F


class MSELoss(Module):
    def __init__(self, reduction: str = 'mean'):
        super(MSELoss, self).__init__()
        if reduction is None or reduction == "":
            reduction = 'mean'

        self.reduction = reduction

    def forward(self, input, target):
        s = torch.square(input - target)
        if self.reduction == "mean":
            return s.mean()
        else:
            return s.sum()


