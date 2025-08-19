from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.Attention import multihead_self_attention
from cs336_basics.SwiGLU import SwiGLU
from torch import nn
import torch

class transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta, max_seq_len, device=None):
        super(transformer_block, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn = multihead_self_attention(d_model, num_heads, theta, max_seq_len)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, token_positions):
        x = self.attn(self.ln1(x), token_positions) + x # attn
        x = self.ffn(self.ln2(x)) + x #ffn
        return x