# ORIGINAL



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




#Static positional token
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
#         self.attn_dropout = nn.Dropout(config.dropout)
#
#         # kqv in one pass
#         self.pos_kq = nn.Linear(config.d_model, 2 * config.d_model, bias=False)
#         self.tok_kqv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
#
#         # safer improvement: output projection
#         self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
#
#         # learn how much token attention and positional attention should contribute
#         self.tok_alpha = nn.Parameter(torch.tensor(1.0))
#         self.pos_alpha = nn.Parameter(torch.tensor(0.3))
#
#         self.relative_bias = config.relative_bias
#         if config.relative_bias:
#             self.bias = nn.Embedding(config.max_len * 2, config.num_heads)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         batch_size, seq_len, _ = x.shape
#
#         pos_embed = self.pos_embed(seq_len).repeat(batch_size, 1, 1)
#         pos_key, pos_query = self.pos_kq(pos_embed).chunk(2, dim=-1)
#
#         pos_key = pos_key.view(batch_size, seq_len, self.num_heads, -1).permute(
#             0, 2, 3, 1
#         )
#         pos_query = pos_query.view(batch_size, seq_len, self.num_heads, -1).transpose(
#             1, 2
#         )
#         pos_attn = torch.matmul(pos_query, pos_key)
#
#         tok_key, tok_query, tok_value = self.tok_kqv(x).chunk(3, dim=-1)
#         tok_key = tok_key.view(batch_size, seq_len, self.num_heads, -1).permute(
#             0, 2, 3, 1
#         )
#         tok_query = tok_query.view(batch_size, seq_len, self.num_heads, -1).transpose(
#             1, 2
#         )
#         tok_value = tok_value.view(batch_size, seq_len, self.num_heads, -1).transpose(
#             1, 2
#         )
#         tok_attn = torch.matmul(tok_query, tok_key)
#
#         #attn = (self.tok_alpha * tok_attn + self.pos_alpha * pos_attn) / self.scale
#         alpha_sum = self.tok_alpha + self.pos_alpha
#         attn = (self.tok_alpha * tok_attn + self.pos_alpha * pos_attn) / (alpha_sum * self.scale)
#
#         if self.relative_bias:
#             relative_positions = get_relative_positions(
#                 seq_len, self.bidirectional, self.num_buckets, self.max_distance
#             ).to(attn.device)
#
#             bias = self.bias(relative_positions + self.max_len)
#             bias = bias.permute(2, 0, 1).unsqueeze(0)
#             attn = attn + bias
#
#         attn = F.softmax(attn, dim=-1)
#         attn = self.attn_dropout(attn)
#
#         out = torch.matmul(attn, tok_value)
#         out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
#         out = self.out_proj(out)
#         out = self.dropout(out)
#         return out


#Gated Positional token
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
#     """
#     Patched TUPE attention for time-series forecasting.
#
#     Changes:
#     - learnable mix between token attention and positional attention
#     - learnable positional strength
#     - local temporal bias to prefer nearer timestamps
#     - recency bias to prefer recent positions
#     """
#
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
#         self.pos_kq = nn.Linear(config.d_model, 2 * config.d_model, bias=False)
#         self.tok_kqv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
#
#         self.relative_bias = getattr(config, "relative_bias", False)
#         if self.relative_bias:
#             self.bias = nn.Embedding(self.num_buckets, self.num_heads)
#
#         self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
#
#         # ---- Added patches ----
#         # learnable mixing between token attention and position attention
#         self.tok_alpha = nn.Parameter(torch.tensor(1.0))
#         self.pos_alpha = nn.Parameter(torch.tensor(0.5))
#
#         # extra learnable position strength
#         self.pos_strength = nn.Parameter(torch.tensor(0.2))
#
#         # local temporal bias strength
#         self.local_bias_strength = nn.Parameter(torch.tensor(0.15))
#
#         # recency bias strength
#         self.recency_bias_strength = nn.Parameter(torch.tensor(0.10))
#
#         # optional clamp range for stability
#         self.alpha_min = 0.05
#         self.alpha_max = 3.0
#
#     def _shape(self, x: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
#         # (B, L, D) -> (B, H, L, Dh)
#         return x.view(bsz, seq_len, self.num_heads, -1).transpose(1, 2).contiguous()
#
#     def _get_relative_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
#         rel_pos = get_relative_positions(
#             seq_len,
#             bidirectional=self.bidirectional,
#             num_buckets=self.num_buckets,
#             max_distance=self.max_distance,
#         ).to(device)  # (L, L)
#         values = self.bias(rel_pos)  # (L, L, H)
#         return values.permute(2, 0, 1).contiguous()  # (H, L, L)
#
#     def _build_local_temporal_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
#         """
#         Penalize far-away positions so nearby time steps are preferred.
#         Shape: (1, 1, L, L)
#         """
#         idx = torch.arange(seq_len, device=device)
#         dist = (idx[None, :] - idx[:, None]).abs().float()  # (L, L)
#
#         # normalized negative distance bias
#         dist = dist / max(seq_len - 1, 1)
#         local_bias = -dist * self.local_bias_strength
#         return local_bias.unsqueeze(0).unsqueeze(0)
#
#     def _build_recency_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
#         """
#         Slightly favor more recent keys.
#         Shape: (1, 1, 1, L)
#         """
#         pos = torch.arange(seq_len, device=device).float()
#         pos = pos / max(seq_len - 1, 1)
#         recency = pos * self.recency_bias_strength
#         return recency.view(1, 1, 1, seq_len)
#
#     def forward(
#         self,
#         x: torch.Tensor,
#         attention_mask: torch.Tensor = None,
#     ) -> torch.Tensor:
#         """
#         x: (B, L, D)
#         attention_mask:
#             optional mask broadcastable to (B, 1, L, L)
#             masked positions should contain large negative values
#         """
#         bsz, seq_len, dim = x.size()
#         device = x.device
#
#         # token projections
#         tok_qkv = self.tok_kqv(x)  # (B, L, 3D)
#         tok_q, tok_k, tok_v = tok_qkv.chunk(3, dim=-1)
#
#         tok_q = self._shape(tok_q, seq_len, bsz)  # (B, H, L, Dh)
#         tok_k = self._shape(tok_k, seq_len, bsz)
#         tok_v = self._shape(tok_v, seq_len, bsz)
#
#         # positional projections
#         pos = self.pos_embed(seq_len).to(device)  # expected (L, D)
#         pos_kq = self.pos_kq(pos)  # (L, 2D)
#         pos_k, pos_q = pos_kq.chunk(2, dim=-1)
#
#         pos_q = pos_q.view(seq_len, self.num_heads, -1).transpose(0, 1).contiguous()  # (H, L, Dh)
#         pos_k = pos_k.view(seq_len, self.num_heads, -1).transpose(0, 1).contiguous()  # (H, L, Dh)
#
#         # token attention
#         tok_attn = torch.matmul(tok_q, tok_k.transpose(-1, -2))  # (B, H, L, L)
#
#         # positional attention
#         pos_attn = torch.matmul(pos_q, pos_k.transpose(-1, -2))  # (H, L, L)
#         pos_attn = pos_attn.unsqueeze(0)  # (1, H, L, L)
#
#         # stabilize learnable weights
#         tok_alpha = torch.clamp(self.tok_alpha, self.alpha_min, self.alpha_max)
#         pos_alpha = torch.clamp(self.pos_alpha, self.alpha_min, self.alpha_max)
#         pos_strength = torch.clamp(self.pos_strength, 0.0, 2.0)
#
#         # combine token and positional attention
#         attn = (tok_alpha * tok_attn + (pos_alpha + pos_strength) * pos_attn) / self.scale
#
#         # add relative bias if enabled
#         if self.relative_bias:
#             rel_bias = self._get_relative_bias(seq_len, device)  # (H, L, L)
#             attn = attn + rel_bias.unsqueeze(0)
#
#         # add local temporal bias
#         attn = attn + self._build_local_temporal_bias(seq_len, device)
#
#         # add recency bias
#         attn = attn + self._build_recency_bias(seq_len, device)
#
#         # external attention mask
#         if attention_mask is not None:
#             attn = attn + attention_mask
#
#         attn = F.softmax(attn, dim=-1)
#         attn = self.dropout(attn)
#
#         out = torch.matmul(attn, tok_v)  # (B, H, L, Dh)
#         out = out.transpose(1, 2).contiguous().view(bsz, seq_len, dim)  # (B, L, D)
#         out = self.o_proj(out)
#         return out

import math
import torch
from torch import nn
from torch.nn import functional as F

from tupe.bias import get_relative_positions
from tupe.config import TUPEConfig


class TUPEMultiHeadAttention(nn.Module):
    def __init__(self, config: TUPEConfig, pos_embed: nn.Module) -> None:
        super().__init__()
        self.max_len = config.max_len
        self.num_heads = config.num_heads
        self.num_buckets = config.num_buckets
        self.max_distance = config.max_distance
        self.bidirectional = getattr(config, "bidirectional_bias", False)
        self.scale = math.sqrt(2 * config.d_head)

        self.pos_embed = pos_embed
        self.relative_bias = getattr(config, "relative_bias", True)

        self.dropout = nn.Dropout(config.dropout)
        self.attn_dropout = nn.Dropout(config.dropout)

        # token and position projections
        self.pos_kq = nn.Linear(config.d_model, 2 * config.d_model, bias=False)
        self.tok_kqv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)

        # per-head gating instead of single fixed sum
        self.tok_gate = nn.Parameter(torch.ones(self.num_heads))
        self.pos_gate = nn.Parameter(torch.ones(self.num_heads) * 0.5)

        # learnable locality strength
        self.local_strength = nn.Parameter(torch.tensor(0.25))

        # output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # local temporal branch
        self.local_conv = nn.Conv1d(
            in_channels=config.d_model,
            out_channels=config.d_model,
            kernel_size=3,
            padding=1,
            groups=config.d_model,
            bias=False,
        )

        if self.relative_bias:
            # signed relative bias, not max_len*2 indexing
            self.bias = nn.Embedding(self.num_buckets, self.num_heads)

    def _shape(self, x: torch.Tensor, seq_len: int, batch_size: int) -> torch.Tensor:
        return x.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2).contiguous()

    def _get_rel_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        rel = get_relative_positions(
            seq_len=seq_len,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        ).to(device)  # (L, L)
        bias = self.bias(rel)  # (L, L, H)
        return bias.permute(2, 0, 1).unsqueeze(0)  # (1, H, L, L)

    def _get_local_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(seq_len, device=device)
        dist = (idx[None, :] - idx[:, None]).abs().float()
        dist = dist / max(seq_len - 1, 1)
        local_bias = -self.local_strength * dist
        return local_bias.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        device = x.device

        # -------- local temporal branch --------
        local_feat = self.local_conv(x.transpose(1, 2)).transpose(1, 2)  # (B, L, D)

        # -------- token attention --------
        tok_k, tok_q, tok_v = self.tok_kqv(x).chunk(3, dim=-1)
        tok_q = self._shape(tok_q, seq_len, batch_size)                  # (B, H, L, Dh)
        tok_k = self._shape(tok_k, seq_len, batch_size)                  # (B, H, L, Dh)
        tok_v = self._shape(tok_v, seq_len, batch_size)                  # (B, H, L, Dh)

        tok_attn = torch.matmul(tok_q, tok_k.transpose(-1, -2))          # (B, H, L, L)

        # -------- positional attention --------
        pos = self.pos_embed(seq_len).to(device)                         # (1, L, D) or (L, D)
        if pos.dim() == 3:
            pos = pos.squeeze(0)                                         # (L, D)

        pos_k, pos_q = self.pos_kq(pos).chunk(2, dim=-1)
        pos_q = pos_q.view(seq_len, self.num_heads, -1).transpose(0, 1)  # (H, L, Dh)
        pos_k = pos_k.view(seq_len, self.num_heads, -1).transpose(0, 1)  # (H, L, Dh)

        pos_attn = torch.matmul(pos_q, pos_k.transpose(-1, -2)).unsqueeze(0)  # (1, H, L, L)

        # -------- per-head gated mixing --------
        tok_gate = torch.sigmoid(self.tok_gate).view(1, self.num_heads, 1, 1)
        pos_gate = torch.sigmoid(self.pos_gate).view(1, self.num_heads, 1, 1)

        attn = (tok_gate * tok_attn + pos_gate * pos_attn) / self.scale

        if self.relative_bias:
            attn = attn + self._get_rel_bias(seq_len, device)

        attn = attn + self._get_local_bias(seq_len, device)

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, tok_v)                                  # (B, H, L, Dh)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.out_proj(out)

        # fuse with local branch
        out = out + local_feat
        out = self.dropout(out)
        return out