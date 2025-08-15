from torch import nn
import torch

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        '''
        RoPE module. Applies rotary embeddings to input tensors.

        Args:
            theta (float): Base frequency θ for RoPE (e.g., 10000)
            d_k (int): Dimension of query/key vectors (must be even)
            max_seq_len (int): Maximum sequence length
        '''
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even"
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self._create_buffers()

    def _create_buffers(self):
        # Frequency terms: θ_i = θ^(-2i / d_k), i ∈ [0, d_k//2)
        i = torch.arange(0, self.d_k // 2).float()
        theta = self.theta ** (-2 * i / self.d_k)  # (d_k//2,)

        # Position indices: m ∈ [0, 1, ..., max_seq_len-1]
        m = torch.arange(self.max_seq_len)  # (seq_len,)

        # Outer product: m * θ_i → (seq_len, d_k//2)
        angles = torch.outer(m, theta)  # (seq_len, d_k//2)

        # Apply sin/cos
        cos_table = torch.cos(angles)  # (seq_len, d_k//2)
        sin_table = torch.sin(angles)  # (seq_len, d_k//2)

        # Register as buffers (not learnable, saved in state_dict if persistent=True)
        self.register_buffer("cos_table", cos_table, persistent=False)
        self.register_buffer("sin_table", sin_table, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to input tensor.

        Args:
            x: (batch_size, seq_len, d_k)
            token_positions: (batch_size, seq_len), integer positions

        Returns:
            Tensor: same shape as x, with rotary embeddings applied
        """
        # Ensure token_positions is long
        if token_positions.dtype not in [torch.long, torch.int]:
            token_positions = token_positions.long()

        # Extract cos/sin for required positions
        cos = self.cos_table[token_positions]  # (batch_size, seq_len, d_k//2)
        sin = self.sin_table[token_positions]  # (batch_size, seq_len, d_k//2)

        # Split x into even and odd indices
        x_even = x[..., 0::2]  # (batch_size, seq_len, d_k//2)
        x_odd  = x[..., 1::2]

        # Apply rotation: [cos * x_even - sin * x_odd, sin * x_even + cos * x_odd]
        x_rotated_even = cos * x_even - sin * x_odd
        x_rotated_odd  = sin * x_even + cos * x_odd

        # Concatenate back
        result = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)  # (..., 2)
        result = result.view_as(x)  # Reshape to (batch_size, seq_len, d_k)

        return result