# import math
#
# import torch
# from torch import nn
# from torch.nn import functional as F
#
# from tupe.bias import get_relative_positions
# from tupe.config import TUPEConfig
#
#
# class TUPEMultiHeadAttention(nn.Module):
#     def __init__(self, config: TUPEConfig, pos_embed: nn.Module) -> None:
#         super().__init__()
#         self.max_len = config.max_len
#         self.num_heads = config.num_heads
#         self.num_buckets = config.num_buckets
#         self.max_distance = config.max_distance
#         self.bidirectional = config.bidirectional_bias
#         self.scale = math.sqrt(2 * config.d_head)
#
#         self.pos_embed = pos_embed
#         self.dropout = nn.Dropout(config.dropout)
#
#         # kqv in one pass
#         self.pos_kq = nn.Linear(config.d_model, 2 * config.d_model, bias=False)
#         self.tok_kqv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
#
#         self.relative_bias = config.relative_bias
#         if config.relative_bias:
#             self.bias = nn.Embedding(config.max_len * 2, config.num_heads)
#
#     def forward(self, x: torch.tensor) -> torch.tensor:
#         batch_size, seq_len, _ = x.shape
#
#         pos_embed = self.pos_embed(seq_len).repeat(batch_size, 1, 1)
#         # pos_embed.shape == (batch_size, seq_len, d_model)
#         pos_key, pos_query = self.pos_kq(pos_embed).chunk(2, dim=-1)
#         pos_key = pos_key.view(batch_size, seq_len, self.num_heads, -1).permute(
#             0, 2, 3, 1
#         )
#         # pos_key.shape == (batch_size, num_heads, d_head, seq_len)
#         pos_query = pos_query.view(batch_size, seq_len, self.num_heads, -1).transpose(
#             1, 2
#         )
#         # pos_query.shape == (batch_size, num_heads, seq_len, d_head)
#         pos_attn = torch.matmul(pos_query, pos_key)
#         # pos_attn.shape == (batch_size, num_heads, seq_len, seq_len)
#
#         tok_key, tok_query, tok_value = self.tok_kqv(x).chunk(3, dim=-1)
#         tok_key = tok_key.view(batch_size, seq_len, self.num_heads, -1).permute(
#             0, 2, 3, 1
#         )
#         # tok_key.shape == (batch_size, num_heads, d_head, seq_len)
#         tok_query = tok_query.view(batch_size, seq_len, self.num_heads, -1).transpose(
#             1, 2
#         )
#         tok_value = tok_value.view(batch_size, seq_len, self.num_heads, -1).transpose(
#             1, 2
#         )
#         # tok_qv.shape == (batch_size, num_heads, seq_len, d_head)
#         tok_attn = torch.matmul(tok_query, tok_key)
#         # tok_attn.shape == (batch_size, num_heads, seq_len, seq_len)
#
#         attn = (tok_attn + pos_attn) / self.scale
#         if self.relative_bias:
#             relative_positions = get_relative_positions(
#                 seq_len, self.bidirectional, self.num_buckets, self.max_distance
#             ).to(attn.device)
#             # relative_positions.shape == (seq_len, seq_len)
#             bias = self.bias(relative_positions + self.max_len)
#             # bias.shape == (seq_len, seq_len, num_heads)
#             bias = bias.permute(2, 0, 1).unsqueeze(0)
#             # bias.shape == (1, num_heads, seq_len, seq_len)
#             attn = attn + bias
#
#         attn = F.softmax(attn, dim=-1)
#         # attn.shape == (batch_size, num_heads, seq_len, seq_len)
#         out = torch.matmul(attn, tok_value)
#         # out.shape == (batch_size, num_heads, seq_len, d_head)
#         out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
#         # out.shape == (batch_size, seq_len, d_model)
#         out = self.dropout(out)
#         return out
import math

import torch
from torch import nn
from torch.nn import functional as F

from tupe.bias import get_relative_positions
from tupe.config import TUPEConfig


class TUPEMultiHeadAttention(nn.Module):
    def __init__(
        self,
        config: TUPEConfig,
        pos_embed: nn.Module,
        causal: bool = True,
        local_window_size: int | None = None,
        recency_bias_strength: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_len = config.max_len
        self.num_heads = config.num_heads
        self.num_buckets = config.num_buckets
        self.max_distance = config.max_distance
        self.bidirectional = config.bidirectional_bias
        self.scale = math.sqrt(2 * config.d_head)

        self.pos_embed = pos_embed
        self.dropout = nn.Dropout(config.dropout)

        # New options
        self.causal = causal
        self.local_window_size = local_window_size
        self.recency_bias_strength = recency_bias_strength

        # kqv in one pass
        self.pos_kq = nn.Linear(config.d_model, 2 * config.d_model, bias=False)
        self.tok_kqv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)

        self.relative_bias = config.relative_bias
        if config.relative_bias:
            self.bias = nn.Embedding(config.max_len * 2, config.num_heads)

    def _build_attention_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor | None:
        """
        Returns a boolean mask of shape (seq_len, seq_len).
        True means 'mask this position out'.
        """
        mask = None

        # Causal mask: block attention to future tokens
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                diagonal=1,
            )
            mask = causal_mask if mask is None else (mask | causal_mask)

        # Local window mask: allow attention only within a recent window
        if self.local_window_size is not None:
            idx = torch.arange(seq_len, device=device)
            distance = idx.unsqueeze(1) - idx.unsqueeze(0)
            # For query i and key j, keep only j in [i-window+1, i]
            local_mask = (distance < 0) | (distance >= self.local_window_size)
            mask = local_mask if mask is None else (mask | local_mask)

        return mask

    def _build_recency_bias(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor | None:
        """
        Returns additive bias of shape (1, 1, seq_len, seq_len).
        Larger values are given to more recent past positions.
        """
        if self.recency_bias_strength <= 0:
            return None

        idx = torch.arange(seq_len, device=device)
        distance = idx.unsqueeze(1) - idx.unsqueeze(0)  # i - j

        # Only past/current tokens receive recency preference
        # More recent => smaller distance => larger bias
        recency = -distance.float()

        # Future positions should not receive positive help
        recency = torch.where(distance >= 0, recency, torch.full_like(recency, -1e4))

        recency = self.recency_bias_strength * recency
        return recency.to(dtype=dtype).unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        pos_embed = self.pos_embed(seq_len).repeat(batch_size, 1, 1)
        # pos_embed.shape == (batch_size, seq_len, d_model)

        pos_key, pos_query = self.pos_kq(pos_embed).chunk(2, dim=-1)
        pos_key = pos_key.view(batch_size, seq_len, self.num_heads, -1).permute(
            0, 2, 3, 1
        )
        # pos_key.shape == (batch_size, num_heads, d_head, seq_len)

        pos_query = pos_query.view(batch_size, seq_len, self.num_heads, -1).transpose(
            1, 2
        )
        # pos_query.shape == (batch_size, num_heads, seq_len, d_head)

        pos_attn = torch.matmul(pos_query, pos_key)
        # pos_attn.shape == (batch_size, num_heads, seq_len, seq_len)

        tok_key, tok_query, tok_value = self.tok_kqv(x).chunk(3, dim=-1)
        tok_key = tok_key.view(batch_size, seq_len, self.num_heads, -1).permute(
            0, 2, 3, 1
        )
        # tok_key.shape == (batch_size, num_heads, d_head, seq_len)

        tok_query = tok_query.view(batch_size, seq_len, self.num_heads, -1).transpose(
            1, 2
        )
        tok_value = tok_value.view(batch_size, seq_len, self.num_heads, -1).transpose(
            1, 2
        )
        # tok_query/tok_value.shape == (batch_size, num_heads, seq_len, d_head)

        tok_attn = torch.matmul(tok_query, tok_key)
        # tok_attn.shape == (batch_size, num_heads, seq_len, seq_len)

        attn = (tok_attn + pos_attn) / self.scale

        if self.relative_bias:
            relative_positions = get_relative_positions(
                seq_len, self.bidirectional, self.num_buckets, self.max_distance
            ).to(attn.device)
            # relative_positions.shape == (seq_len, seq_len)

            bias = self.bias(relative_positions + self.max_len)
            # bias.shape == (seq_len, seq_len, num_heads)

            bias = bias.permute(2, 0, 1).unsqueeze(0)
            # bias.shape == (1, num_heads, seq_len, seq_len)

            attn = attn + bias

        # Add recency bias
        recency_bias = self._build_recency_bias(
            seq_len=seq_len,
            device=attn.device,
            dtype=attn.dtype,
        )
        if recency_bias is not None:
            attn = attn + recency_bias

        # Apply mask
        mask = self._build_attention_mask(seq_len=seq_len, device=attn.device)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        # attn.shape == (batch_size, num_heads, seq_len, seq_len)

        out = torch.matmul(attn, tok_value)
        # out.shape == (batch_size, num_heads, seq_len, d_head)

        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)

        out = self.dropout(out)
        return out
