import torch
from torch import nn
import einops as eo


class VanillaMultiHeadSelfAttention(nn.Module):
    def __init__(self, qk_dim, v_dim, num_heads):
        self.qk_dim = qk_dim
        self.v_dim = v_dim

        assert qk_dim % num_heads == 0
        assert v_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.qk_dim_per_head = qk_dim // num_heads
        self.v_dim_per_head = v_dim // num_heads

        self.Q = nn.Linear(qk_dim, qk_dim)
        self.K = nn.Linear(qk_dim, qk_dim)
        self.V = nn.Linear(v_dim, v_dim)
        self.out_proj = nn.Linear(v_dim, v_dim) # this is the output projection layer

    def forward(self, x):
        # TODO: add mask later here...

        B, T, D = x.shape

        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        qi = q.view(B, T, self.num_heads, self.qk_dim_per_head)
        ki = k.view(B, T, self.num_heads, self.qk_dim_per_head)
        vi = v.view(B, T, self.num_heads, self.v_dim_per_head)

        attn_mat = eo.einsum(qi, ki, "b tq h d, b tk h d -> b h tq tk")
        attn_mat = attn_mat / (self.qk_dim_per_head ** 0.5)

        attn_mat = torch.softmax(attn_mat, dim=-1)

        attn_head_outputs = eo.einsum(attn_mat, vi, "b h tq tk, b tk h d -> b tq (h d)")
        attn_out_proj = self.out_proj(attn_head_outputs)

        return attn_out_proj
