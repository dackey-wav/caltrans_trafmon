"""TimesNet forecaster for the station channels.

Input: the last ``seq_len`` steps of the target channels plus time marks that
cover the input window and the forecast horizon. Calendar flags are known in
advance, so the horizon part of the marks carries holidays and school breaks
into the prediction. Output: the next ``pred_len`` steps of every channel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.TimesNet.config import (
    DEFAULT_D_FF,
    DEFAULT_D_MODEL,
    DEFAULT_DROPOUT,
    DEFAULT_E_LAYERS,
    DEFAULT_NUM_KERNELS,
    DEFAULT_TOP_K,
)
from models.TimesNet.layers.Conv_Blocks import Inception_Block_V1
from models.TimesNet.layers.Embed import PositionalEmbedding, TokenEmbedding


@dataclass
class TimesNetConfig:
    seq_len: int = 192
    pred_len: int = 96
    enc_in: int = 3
    c_out: int = 3
    n_mark: int = 5
    e_layers: int = DEFAULT_E_LAYERS
    d_model: int = DEFAULT_D_MODEL
    d_ff: int = DEFAULT_D_FF
    top_k: int = DEFAULT_TOP_K
    num_kernels: int = DEFAULT_NUM_KERNELS
    dropout: float = DEFAULT_DROPOUT
    residual: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def FFT_for_Period(x, k=2):
    # cuFFT in half precision only accepts power-of-two lengths.
    xf = torch.fft.rfft(x.float(), dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    top_list = np.maximum(top_list, 1)
    period = np.clip(x.shape[1] // top_list, 1, max(x.shape[1], 1))
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs: TimesNetConfig):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(
                configs.d_model, configs.d_ff, num_kernels=configs.num_kernels
            ),
            nn.GELU(),
            Inception_Block_V1(
                configs.d_ff, configs.d_model, num_kernels=configs.num_kernels
            ),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros(
                    [x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]],
                    dtype=x.dtype,
                    device=x.device,
                )
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, : (self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1).to(res.dtype)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        return res + x


class Model(nn.Module):
    """Direct multi-step TimesNet forecaster."""

    def __init__(self, configs: TimesNetConfig):
        super().__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.value_embedding = TokenEmbedding(c_in=configs.enc_in, d_model=configs.d_model)
        self.position_embedding = PositionalEmbedding(d_model=configs.d_model)
        self.mark_embedding = nn.Linear(configs.n_mark, configs.d_model, bias=False)
        self.dropout = nn.Dropout(configs.dropout)
        self.predict_linear = nn.Linear(self.seq_len, self.seq_len + self.pred_len)
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_enc: torch.Tensor, x_mark: torch.Tensor) -> torch.Tensor:
        """x_enc: (B, seq_len, C). x_mark: (B, seq_len + pred_len, n_mark)."""
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        enc = self.value_embedding(x)
        enc = self.predict_linear(enc.permute(0, 2, 1)).permute(0, 2, 1)
        enc = enc + self.position_embedding(enc) + self.mark_embedding(x_mark)
        enc = self.dropout(enc)
        for block in self.model:
            enc = self.layer_norm(block(enc))
        out = self.projection(enc)
        out = out[:, -self.pred_len :, :]
        if self.configs.residual:
            # The target is a correction to last week, so it lives in the
            # normalized space and must not be shifted back to the input level.
            return out
        return out * stdev + means


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
