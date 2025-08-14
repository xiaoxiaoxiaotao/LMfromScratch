from einops import einsum
import torch
from torch import nn
import math


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        '''
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        super(Embedding, self).__init__()
        self.weights = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)).to(device)
        self._init_weights()
    
    def _init_weights(self):
        std = (1.0 / math.sqrt(self.weights.size(1))) * 0.5 
        torch.nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        #  (batch_size, sequence_lenth) -> (batch_size, sequence_lenth, d_model)
        return self.weights[token_ids]
