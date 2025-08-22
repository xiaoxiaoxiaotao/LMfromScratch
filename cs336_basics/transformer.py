from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.Attention import multihead_self_attention
from cs336_basics.SwiGLU import SwiGLU
from torch import nn
import torch
from .Embedding import Embedding
from typing import Dict, List, Tuple, Optional, Iterable, Iterator

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
        y = self.attn(self.ln1(x), token_positions) + x # attn
        output = self.ffn(self.ln2(y)) + y #ffn
        return output

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        theta: float = 10000.0,
        device=None
    ):
        """
        Full Transformer Language Model.
        
        Args:
            vocab_size: Size of the vocabulary.
            context_length: Maximum sequence length (for pos embedding or RoPE caching).
            d_model: Embedding dimension.
            num_heads: Number of attention heads.
            d_ff: Dimension of the feed-forward network.
            num_layers: Number of Transformer blocks.
            theta: Rotary embedding base (default 10000.0).
            device: Device to place parameters on.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.device = device

        # Token embeddings
        self.token_emb = Embedding(vocab_size, d_model, device=device)

        # Transformer blocks
        self.layers = nn.ModuleList([
            transformer_block(d_model, num_heads, d_ff, theta, context_length, device)
            for _ in range(num_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(d_model)

        # Unembedding / output head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, device=device)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights as per standard practice."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Truncated normal initialization
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.LongTensor, token_positions: Optional[torch.LongTensor] = None):
        """
        Forward pass of the Transformer language model.
        
        Args:
            input_ids: (B, T) tensor of token indices.
            token_positions: (B, T) tensor of position indices for RoPE.
                             If None, defaults to cumulative sequence positions.
        
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.shape
        device = input_ids.device
        self.device = device

        # Generate token positions if not provided
        if token_positions is None:
            token_positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

        # Token embeddings
        x = self.token_emb(input_ids)  # (B, T, d_model)

        # Forward through transformer blocks
        for layer in self.layers:
            x = layer(x, token_positions)

        # Final layer norm
        x = self.norm(x)

        # Output logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits