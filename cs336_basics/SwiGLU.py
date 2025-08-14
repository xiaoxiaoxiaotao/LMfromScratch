from torch import nn
import torch
from cs336_basics.Linear import Linear

def silu(x):
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, dff: int, device=None, dtype=None):
        super().__init__()
        self.W1 = Linear(d_model, dff)
        self.W2 = Linear(dff, d_model)
        self.W3 = Linear(d_model, dff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # W2(SiLU(W1x) ⊙ W3x)
        # x (batch_size, seq_len, d_model)
        # W1x/W3x (batch_size, seq_len, d_model) * (d_model, dff) -> (batch_size, seq_len, dff)
        W1x = self.W1(x)
        W3x = self.W3(x)
        medium = silu(W1x) * W3x # (batch_size, seq_len, dff)
        return self.W2(medium)
        