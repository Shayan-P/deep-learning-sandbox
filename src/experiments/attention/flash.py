import torch
from torch import nn
import einops as eo
from dataclasses import dataclass
from typing import Optional

# TODO: later explore how the causal masking would affect the implementation of the streaming softmax...

@dataclass
class StreamingSoftmaxState:
    max_score: float
    denom_normalized_scores: float
    running_value: torch.Tensor # (D, )

    def get_updated_state(self, score: float, value: torch.Tensor) -> "StreamingSoftmaxState":
        new_max_score = max(self.max_score, score)
        new_max_delta = new_max_score - self.max_score
        this_normalized_score = score - new_max_score

        exp_neg_new_max_delta = torch.exp(-new_max_delta)
        exp_this_normalized_score = torch.exp(this_normalized_score)

        new_denom_normalized_scores = self.denom_normalized_scores * exp_neg_new_max_delta + exp_this_normalized_score
        denom_change_ratio = new_denom_normalized_scores / self.denom_normalized_scores

        return StreamingSoftmaxState(
            max_score=new_max_score,
            denom_normalized_scores=new_denom_normalized_scores,
            running_value=denom_change_ratio * self.running_value + value * exp_this_normalized_score / new_denom_normalized_scores,
        )


@dataclass
class StreamingSoftmaxStateTensor:
    """
    Similar to StreamingSoftmaxState, but a score block (B, T) and a tensor block (B, T, D)
    """

    max_score: torch.Tensor # (B,)
    denom_normalized_scores: torch.Tensor # (B,)
    running_value: torch.Tensor # (B, D)

    @staticmethod
    def initialize(score: torch.Tensor, value: torch.Tensor) -> "StreamingSoftmaxStateTensor":
        B, T = score.shape
        assert len(value.shape) == 3
        D = value.shape[2]
        assert value.shape == (B, T, D)

        max_score = score.max(dim=1).values
        normalized_scores = score - max_score[:, None]
        denom_normalized_scores = torch.exp(normalized_scores).sum(dim=1)
        running_value = (value * normalized_scores[:, :, None]).sum(dim=1) / denom_normalized_scores[:, None]

        return StreamingSoftmaxStateTensor(
            max_score=max_score,
            denom_normalized_scores=denom_normalized_scores,
            running_value=running_value,
        )

    def get_updated_state(self, score: float, value: torch.Tensor) -> "StreamingSoftmaxStateTensor":
        B, T = score.shape
        assert len(value.shape) == 3
        D = value.shape[2]
        assert value.shape == (B, T, D)

        new_max_score = torch.maximum(self.max_score, score.max(dim=-1).values)
        new_max_delta = new_max_score - self.max_score
        this_normalized_score = score - new_max_score[:, None]

        exp_neg_new_max_delta = torch.exp(-new_max_delta)
        exp_this_normalized_score = torch.exp(this_normalized_score)

        new_denom_normalized_scores = self.denom_normalized_scores * exp_neg_new_max_delta + exp_this_normalized_score.sum(dim=-1)
        denom_change_ratio = new_denom_normalized_scores / self.denom_normalized_scores

        return StreamingSoftmaxStateTensor(
            max_score=new_max_score,
            denom_normalized_scores=new_denom_normalized_scores,
            running_value=(
                denom_change_ratio[:, None] * self.running_value + \
                (value * exp_this_normalized_score[:, :, None]).sum(dim=1) / new_denom_normalized_scores[:, None]
            ),
        )


@dataclass
class StreamingKVCacheState:
    """
    Similar to StreamingSoftmaxState, but a score block (B, T) and a tensor block (B, T, D)
    """

    max_score: torch.Tensor # (B, T)
    denom_normalized_scores: torch.Tensor # (B, T)
    k_cache: torch.Tensor # (B, T, D)
    v_cache: torch.Tensor # (B, T, D)

    @staticmethod
    def concatenate(s1: "StreamingKVCacheState", s2: "StreamingKVCacheState") -> "StreamingKVCacheState":
        """
        concatenate along the token dimension...
        """

        return StreamingKVCacheState(
            max_score=torch.concat([s1.max_score, s2.max_score], dim=1),
            denom_normalized_scores=torch.concat([s1.denom_normalized_scores, s2.denom_normalized_scores], dim=1),
            k_cache=torch.concat([s1.k_cache, s2.k_cache], dim=1),
            v_cache=torch.concat([s1.v_cache, s2.v_cache], dim=1),
        )

    @staticmethod
    def initialize(score_mat: torch.Tensor, key: torch.Tensor, value_all: torch.Tensor) -> "StreamingKVCacheState":
        """
        initializes new streams of the kv cache

        score_mat: (B, T_new, T_total)
        key: (B, T_new, D_k)
        value_all: (B, T_total, D_v)
        """
        B, T_new, T_total = score_mat.shape
        B, T_new, D_k = key.shape
        B, T_total, D_v = value_all.shape

        assert score_mat.shape == (B, T_new, T_total)
        assert key.shape == (B, T_new, D_k)
        assert value_all.shape == (B, T_total, D_v)

        max_score = score_mat.max(dim=-1).values
        exp_scores = torch.exp(score_mat - max_score[:, :, None])
        denom_normalized_scores = exp_scores.sum(dim=-1)
        normalized_scores = exp_scores / denom_normalized_scores[:, :, None]
        k_cache = key
        v_cache = eo.einsum(normalized_scores, value_all, "b t_new t_total, b t_total d -> b t_new d")

        return StreamingKVCacheState(
            max_score=max_score,
            denom_normalized_scores=denom_normalized_scores,
            k_cache=k_cache,
            v_cache=v_cache,
        )

    def get_updated_state(self, score_mat: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> "StreamingSoftmaxStateTensor":
        """
        updates the kv caches for the current streams

        score_mat: (B, T_new, T_total)
        key: (B, T_new, D_k)
        value: (B, T_new, D_v)
        """

        B, T_new, T_total = score_mat.shape
        B, T_new, D_k = key.shape
        B, T_new, D_v = value.shape
        B, T_old, D_k = self.k_cache.shape
        B, T_old, D_v = self.v_cache.shape

        assert T_total == T_old + T_new
        assert score_mat.shape == (B, T_new, T_total)
        assert key.shape == (B, T_new, D_k)
        assert value.shape == (B, T_new, D_v)
        assert self.k_cache.shape == (B, T_old, D_k)
        assert self.v_cache.shape == (B, T_old, D_v)

        ### (1) calculate the continuation of the T_old section
        score_mat_old = eo.rearrange(score_mat[:, :, :T_old], "b t_new t_old -> b t_old t_new")
        new_max_score_old = torch.maximum(self.max_score, score_mat_old.max(dim=-1).values)
        new_max_delta_old = new_max_score_old - self.max_score
        this_normalized_score_old = score_mat_old - new_max_score_old[:, :, None]

        exp_neg_new_max_delta_old = torch.exp(-new_max_delta_old)
        exp_this_normalized_score_old = torch.exp(this_normalized_score_old)

        new_denom_normalized_scores_old = self.denom_normalized_scores * exp_neg_new_max_delta_old + exp_this_normalized_score_old.sum(dim=-1)
        denom_change_ratio_old = new_denom_normalized_scores_old / self.denom_normalized_scores

        return StreamingKVCacheState(
            max_score=new_max_score_old,
            denom_normalized_scores=new_denom_normalized_scores_old,
            k_cache=self.k_cache,
            v_cache=(
                denom_change_ratio_old[:, :, None] * self.v_cache + \
                eo.einsum(value, exp_this_normalized_score_old, "b t_new d, b t_old t_new -> b t_old d") / new_denom_normalized_scores_old[:, :, None]
            ),
        )


class FlashAttention(nn.Module):
    def __init__(self, qk_dim, v_dim):
        self.qk_dim = qk_dim
        self.v_dim = v_dim

        self.Q = nn.Linear(qk_dim, qk_dim)
        self.K = nn.Linear(qk_dim, qk_dim)
        self.V = nn.Linear(v_dim, v_dim)
        self.out_proj = nn.Linear(v_dim, v_dim) # this is the output projection layer

    def forward_block(self, streaming_state: Optional[StreamingKVCacheState], x_block: torch.Tensor) -> StreamingKVCacheState:
        """
        for now assume there is no attention mask... in fact with causal mask the problem becomes easier...

        Returns: updated_streaming_state
        """

        B, T_new, D = x_block.shape

        q = self.Q(x_block) # (b tq d)
        v = self.V(x_block) # (b tq d)
        k = self.K(x_block) # (b tq d)

        if streaming_state is None:
            attn_mat = eo.einsum(q, k, "b tq d, b tk d -> b tq tk")
            attn_mat = attn_mat / (self.qk_dim ** 0.5)
            new_streaming_state = StreamingKVCacheState.initialize(
                attn_mat, k, v
            )
            return new_streaming_state

        T_old = streaming_state.k_cache.shape[1]

        k_concat = torch.cat([streaming_state.k_cache, k], dim=1)
        v_concat = torch.cat([streaming_state.v_cache, v], dim=1)

        attn_mat = eo.einsum(q, k_concat, "b tq d, b tk d -> b tq tk")
        attn_mat = attn_mat / (self.qk_dim ** 0.5)

        updated_streaming_state = streaming_state.get_updated_state(
            attn_mat, k, v
        )

        new_streaming_state = StreamingKVCacheState.initialize(
            attn_mat, k, v_concat
        )

        res_streaming_state = StreamingKVCacheState.concatenate(updated_streaming_state, new_streaming_state)
        return res_streaming_state

    def forward(self, x: torch.Tensor, block_size: int) -> torch.Tensor:
        """
        x: (B, T, D)
        block_size: int
        """

        B, T, D = x.shape

        # try out the idea of unequal block sizes... distributed the compute more equally

        num_blocks = (T + block_size - 1) // block_size

        x_blocks = torch.chunk(x, block_size, dim=1)

        streaming_state: Optional[StreamingKVCacheState] = None

        for i, x_block in enumerate(x_blocks):
            streaming_state = self.forward_block(streaming_state, x_block)

        return streaming_state.v_cache
