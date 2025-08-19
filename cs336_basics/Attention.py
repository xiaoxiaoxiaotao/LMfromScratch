import torch
from einops import einsum
from torch import nn
from cs336_basics.Linear import Linear
from cs336_basics.RoPE import RoPE

def softmax(x , dim: int):
    max_x = torch.max(x, dim, keepdim=True)[0]
    exp_x = torch.exp(x-max_x)

    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp

def scaled_dot_product_attention(query,key,value,mask=None):
    # k,q: (batch_size, ..., seq_len, d_k)
    # v: (batch_size, ..., seq_len, d_v)
    # mask: (seq_len, seq_len)
    # return: (batch_size, ..., d_v)
    d_k = key.shape[-1]
    qk = einsum(query,key,
                "batch_size ... seq_len_q d_k, \
                batch_size ... seq_len_k d_k ->  \
                batch_size ... seq_len_q seq_len_k  \
            ") / d_k ** (1/2)
    if mask is not None:
        qk = qk.masked_fill(~mask, float('-inf'))
    propability_qk = softmax(qk, dim=-1)
    result = einsum(propability_qk, value, 
                    "batch_size ... seq_len_q seq_len_k, \
                    batch_size ... seq_len_k d_v -> \
                    batch_size ... seq_len_q d_v")
    return result

def _causal_mask(seq_len: int):
        # [1, 1, seq_len, seq_len] - broadcastable to (B, H, S, S)
        return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

class multihead_self_attention(nn.Module):
    def __init__(self, d_model, num_heads, theta = None, max_seq_len = None):
        super(multihead_self_attention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)

        if (theta is not None) and (max_seq_len is not None):
            self.rope = RoPE(theta,self.head_dim, max_seq_len)
    
    def forward(self,x, token_positions = None, causal = True):
        # x: (*batch_shape, sequence_length d_in)
        *batch_shape, seq_len, _ = x.shape

        Q = self.q_proj(x).view(*batch_shape, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(*batch_shape, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(*batch_shape, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        mask = None
        if causal:
            mask = _causal_mask(seq_len)
        
        if  token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        scores = scaled_dot_product_attention(Q,K,V,mask=mask) # *batch_shape num_heads seq_len_k d_v

        scores = scores.transpose(1, 2).contiguous().view(*batch_shape, seq_len, self.d_model)

        return self.o_proj(scores)