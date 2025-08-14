from torch import nn
import torch
from einops import einsum

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        '''
        RoPE module and create buffers if needed
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        '''
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self._create_buffers()

    def _create_buffers(self):
        # dimention groups (d_k // 2,)
        theta_numerator = torch.arange(0, self.d_k, 2).float()
        theta = 1.0 / (self.theta ** (theta_numerator / self.d_k))  # shape: (dim//2,)

        # position: m: [0, 1, 2, ..., seq_len-1]
        m = torch.arange(self.max_seq_len, device=self.device)  # (seq_len,)

        angles = einsum(theta, m, "i, j -> i j") # (dim//2, seq_len)

        cos_table = torch.cos(angles).to(self.device) # (dim//2, seq_len)
        sin_table = torch.sin(angles).to(self.device) # (dim//2, seq_len)

        self.register_buffer("cos_table", cos_table, persistent=False)
        self.register_buffer("sin_table", sin_table, persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, sequence_length, d_k), token_positions: (batch_size, sequence_length)

        cos = self.cos_table[token_positions] # (batch_size, sequence_length, d_k // 2)
        sin = self.sin_table[token_positions]

        x_front_rot = cos * x[..., :self.d_k//2] - sin * x[..., self.d_k//2:] # (batch_size, sequence_length, d_k // 2)
        x_back_rot  = sin * x[..., :self.d_k//2] + cos * x[..., self.d_k//2:]

        result = torch.cat([x_front_rot, x_back_rot], dim=-1)

        return result

        