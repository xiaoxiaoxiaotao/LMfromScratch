from einops import einsum
import torch
from torch import nn
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype)).to(device)
        self._init_weights()
    
    def _init_weights(self):
        std = (1.0 / math.sqrt(self.weights.size(1))) * 0.5 
        torch.nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape (batch_size, sequence_lenth, d_model/in_features)
        return einsum(x, self.weights, \
                    'batch_size sequence_lenth in_features, \
                    out_features in_features -> \
                    batch_size sequence_lenth out_features'
                )
