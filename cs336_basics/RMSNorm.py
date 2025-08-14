import torch
from torch import nn
import math

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(d_model)).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #Process an input tensor of shape(batch_size, sequence_length, d_model) and return a tensor of the same shape.
        in_dtype = x.dtype
        x = x.to(torch.float32)

        squared_mean = torch.mean(x ** 2, dim=-1, keepdim=True)
        RMS_a = torch.sqrt(squared_mean + self.eps)

        result = (x / RMS_a) * self.weights
        
        # Return the result in the original dtype
        return result.to(in_dtype)