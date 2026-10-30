"""TimesNet model for multivariate anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass

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
    embed_freq_for,
)
from models.TimesNet.layers.Conv_Blocks import Inception_Block_V1
from models.TimesNet.layers.Embed import DataEmbedding


@dataclass
class TimesNetConfig:
    task_name: str = "anomaly_detection"
    seq_len: int = 48
    label_len: int = 0
    pred_len: int = 0
    e_layers: int = DEFAULT_E_LAYERS
    d_model: int = DEFAULT_D_MODEL
    d_ff: int = DEFAULT_D_FF
    top_k: int = DEFAULT_TOP_K
    num_kernels: int = DEFAULT_NUM_KERNELS
    enc_in: int = 10
    c_out: int = 10
    embed: str = "timeF"
    freq: str = "h"
    dropout: float = DEFAULT_DROPOUT
    n_time_mark: int = 4


def build_timesnet_config(
    *,
    n_features: int,
    window: int,
    data_freq: str,
    d_model: int = DEFAULT_D_MODEL,
    d_ff: int = DEFAULT_D_FF,
    e_layers: int = DEFAULT_E_LAYERS,
    top_k: int = DEFAULT_TOP_K,
    num_kernels: int = DEFAULT_NUM_KERNELS,
    dropout: float = DEFAULT_DROPOUT,
) -> TimesNetConfig:
    embed_freq = embed_freq_for(data_freq)
    n_time_mark = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}[embed_freq]
    return TimesNetConfig(
        seq_len=window,
        enc_in=n_features,
        c_out=n_features,
        d_model=d_model,
        d_ff=d_ff,
        e_layers=e_layers,
        top_k=top_k,
        num_kernels=num_kernels,
        freq=embed_freq,
        dropout=dropout,
        n_time_mark=n_time_mark,
    )


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    # Avoid division by zero; clamp period to valid reshape range.
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
                    [x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]
                ).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, : (self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        res = res + x
        return res


class Model(nn.Module):
    """TimesNet anomaly-detection model (reconstruction)."""

    def __init__(self, configs: TimesNetConfig):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len
            )
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name in ("imputation", "anomaly_detection"):
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def anomaly_detection(
        self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None = None
    ) -> torch.Tensor:
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out.mul(
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        dec_out = dec_out.add(
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        return dec_out

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.task_name == "anomaly_detection":
            return self.anomaly_detection(x_enc, x_mark_enc)
        raise ValueError(f"Unsupported task_name: {self.task_name!r}")
