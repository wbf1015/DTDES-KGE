from abc import ABC
from typing import Tuple
import torch
from torch import nn

class Regularizer(nn.Module, ABC):
    def forward(self, factors: Tuple[torch.Tensor]):
        pass
    

class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            for f in factor:
                norm += self.weight * torch.sum(
                    torch.abs(f) ** 3
                ) / f.shape[0]
        
        loss_reocrd = {
            'reg_loss' : norm.item()
        }
        
        return norm, loss_reocrd