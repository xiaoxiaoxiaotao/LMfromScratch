from torch import nn
import torch


class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE) module.
    Applies rotation to queries and keys in self-attention.

    Supports input shapes:
        - (B, T, D): single-head or post-embedding
        - (B, H, T, D): multi-head attention

    All heads share the same rotary embedding (standard practice).
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        """
        Args:
            theta (float): Frequency base (e.g., 10000)
            d_k (int): Dimension of each head (must be even)
            max_seq_len (int): Maximum sequence length for precomputed embeddings
        """
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even"
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self._create_buffers()

    def _create_buffers(self):
        """
        Precompute cos and sin tables of shape (max_seq_len, d_k // 2)
        """
        # Frequency: θ_i = theta^(-2i / d_k) for i in [0, d_k//2)
        i = torch.arange(0, self.d_k // 2).float()  # (d_k//2,)
        freqs = self.theta ** (-2 * i / self.d_k)  # (d_k//2,)

        # Positions: m ∈ [0, 1, ..., max_seq_len-1]
        m = torch.arange(self.max_seq_len)  # (seq_len,)
        angles = torch.outer(m, freqs)  # (seq_len, d_k//2)

        # Register as buffers: (seq_len, d_k//2)
        self.register_buffer("cos_half", torch.cos(angles), persistent=False)
        self.register_buffer("sin_half", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to input tensor.

        Args:
            x: Tensor of shape (B, T, D) or (B, H, T, D), where D == d_k
            token_positions: Long tensor of shape (B, T) or (T,) indicating position ids

        Returns:
            Tensor of same shape as x, with rotary embeddings applied.
        """
        if token_positions.dtype not in [torch.long, torch.int]:
            token_positions = token_positions.long()

        original_shape = x.shape
        if x.dim() == 3:
            # Shape: (B, T, D)
            B, T, D = x.shape
            H = None
        elif x.dim() == 4:
            # Shape: (B, H, T, D)
            B, H, T, D = x.shape
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D: {x.shape}")

        assert D == self.d_k, f"Expected d_k={self.d_k}, got {D}"
        assert D % 2 == 0, "d_k must be even"

        # Get cos/sin from buffer using token positions
        # self.cos_half: (max_seq_len, d_k//2)
        cos = self.cos_half[token_positions]  # (..., T, d_k//2)
        sin = self.sin_half[token_positions]

        # Add head dimension for broadcasting: -> (..., 1, T, d_k//2)
        if cos.dim() == 2:
            # token_positions is (T,) -> cos: (T, d_k//2)
            cos = cos.unsqueeze(0).unsqueeze(1)  # (1, 1, T, d_k//2)
            sin = sin.unsqueeze(0).unsqueeze(1)
        else:
            # token_positions is (B, T) -> cos: (B, T, d_k//2)
            cos = cos.unsqueeze(1)  # (B, 1, T, d_k//2)
            sin = sin.unsqueeze(1)

        # Reshape x to (B, *, T, d_k//2, 2) for rotation
        x_reshaped = x.view(B, -1, T, D // 2, 2)  # (B, *, T, d_k//2, 2)
        x0 = x_reshaped[..., 0]  # real part
        x1 = x_reshaped[..., 1]  # imag part

        # Apply rotation: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
        x0_rot = x0 * cos - x1 * sin
        x1_rot = x0 * sin + x1 * cos

        # Stack back and reshape to original shape
        x_rotated = torch.stack([x0_rot, x1_rot], dim=-1)  # (B, *, T, d_k//2, 2)
        output = x_rotated.view(original_shape)  # restore input shape

        return output