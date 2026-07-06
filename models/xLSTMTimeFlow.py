import math
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            t = t.squeeze(-1)

        half_dim = self.dim // 2
        device = t.device

        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=device).float()
            / max(half_dim - 1, 1)
        )

        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class ConditionalFlowHead(nn.Module):
    """
    v_theta(y_t, t | context)

    y_t:     [B, pred_len]
    t:       [B]
    context: [B, d_model]
    output:  [B, pred_len]
    """

    def __init__(
        self,
        pred_len: int,
        d_model: int,
        time_emb_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)

        in_dim = pred_len + d_model + time_emb_dim

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(
                    dim=hidden_dim,
                    hidden_dim=hidden_dim * 2,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, pred_len)

    def forward(self, y_t, t, context):
        t_emb = self.time_emb(t)
        h = torch.cat([y_t, t_emb, context], dim=-1)

        h = self.input_proj(h)

        for block in self.blocks:
            h = block(h)

        return self.out_proj(self.out_norm(h))


class Model(nn.Module):
    """
    TSLib 风格模型类。

    输入:
        x_enc: [B, seq_len, enc_in]

    训练目标:
        y: [B, pred_len]

    输出:
        pred: [B, pred_len]
    """

    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.num_sampling_steps = getattr(configs, "num_sampling_steps", 20)
        self.num_samples = getattr(configs, "num_samples", 8)

        dropout = getattr(configs, "dropout", 0.1)
        num_blocks = getattr(configs, "e_layers", 4)
        num_heads = getattr(configs, "n_heads", 4)

        flow_hidden_dim = getattr(configs, "flow_hidden_dim", 256)
        flow_layers = getattr(configs, "flow_layers", 3)
        time_emb_dim = getattr(configs, "time_emb_dim", 64)

        slstm_at = getattr(configs, "slstm_at", "")
        if isinstance(slstm_at, str):
            if slstm_at.strip() == "":
                # 建议先只用 mLSTM，更容易跑通
                slstm_at = []
            else:
                slstm_at = [int(x) for x in slstm_at.split(",")]

        self.input_proj = nn.Sequential(
            nn.Linear(self.enc_in, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
            nn.Dropout(dropout),
        )

        xlstm_cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=num_heads,
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla",
                    num_heads=num_heads,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=1.3,
                    act_fn="gelu",
                ),
            ),
            context_length=self.seq_len,
            num_blocks=num_blocks,
            embedding_dim=self.d_model,
            slstm_at=slstm_at,
        )

        self.encoder = xLSTMBlockStack(xlstm_cfg)
        self.context_norm = nn.LayerNorm(self.d_model)

        self.flow_head = ConditionalFlowHead(
            pred_len=self.pred_len,
            d_model=self.d_model,
            time_emb_dim=time_emb_dim,
            hidden_dim=flow_hidden_dim,
            num_layers=flow_layers,
            dropout=dropout,
        )

    def encode_context(self, x_enc):
        """
        x_enc: [B, seq_len, enc_in]
        return:
            context: [B, d_model]
        """

        B, T, C = x_enc.shape

        assert T == self.seq_len, f"Expected seq_len={self.seq_len}, got {T}"
        assert C == self.enc_in, f"Expected enc_in={self.enc_in}, got {C}"

        h = self.input_proj(x_enc)
        h = self.encoder(h)

        context = self.context_norm(h[:, -1, :])

        return context

    def vector_field(self, y_t, t, context):
        return self.flow_head(y_t, t, context)

    def flow_matching_loss(self, x_enc, y):
        """
        Rectified Flow / Conditional Flow Matching loss.

        x_enc: [B, 32, enc_in]
        y:     [B, 4]
        """

        B = y.size(0)
        device = y.device

        context = self.encode_context(x_enc)

        noise = torch.randn_like(y)
        t = torch.rand(B, device=device)

        t_view = t.view(B, 1)

        y_t = (1.0 - t_view) * noise + t_view * y
        target_velocity = y - noise

        pred_velocity = self.vector_field(y_t, t, context)

        loss = F.mse_loss(pred_velocity, target_velocity)

        return loss

    def loss(self, x_enc, y):
        return self.flow_matching_loss(x_enc, y)

    @torch.no_grad()
    def sample_heun(
        self,
        x_enc,
        num_steps: Optional[int] = None,
        num_samples: Optional[int] = None,
        return_all_samples: bool = False,
    ):
        """
        x_enc: [B, 32, enc_in]

        return:
            if return_all_samples:
                samples: [S, B, pred_len]
            else:
                pred_mean: [B, pred_len]
        """

        self.eval()

        if num_steps is None:
            num_steps = self.num_sampling_steps

        if num_samples is None:
            num_samples = self.num_samples

        B = x_enc.size(0)
        device = x_enc.device
        dtype = x_enc.dtype

        context = self.encode_context(x_enc)

        all_samples = []

        for _ in range(num_samples):
            y_t = torch.randn(
                B,
                self.pred_len,
                device=device,
                dtype=dtype,
            )

            dt = 1.0 / num_steps

            for i in range(num_steps):
                t0 = torch.full(
                    (B,),
                    fill_value=i / num_steps,
                    device=device,
                    dtype=dtype,
                )

                t1 = torch.full(
                    (B,),
                    fill_value=(i + 1) / num_steps,
                    device=device,
                    dtype=dtype,
                )

                v0 = self.vector_field(y_t, t0, context)
                y_euler = y_t + dt * v0

                v1 = self.vector_field(y_euler, t1, context)

                y_t = y_t + 0.5 * dt * (v0 + v1)

            all_samples.append(y_t)

        samples = torch.stack(all_samples, dim=0)

        if return_all_samples:
            return samples

        return samples.mean(dim=0)

    @torch.no_grad()
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        TSLib 常见 forward 接口。
        推理时返回点预测均值。

        x_enc: [B, 32, enc_in]
        return:
            pred: [B, 4]
        """
        return self.sample_heun(
            x_enc,
            num_steps=self.num_sampling_steps,
            num_samples=self.num_samples,
            return_all_samples=False,
        )

    @torch.no_grad()
    def anomaly_score_per_horizon(
        self,
        x_enc,
        y,
        num_steps: Optional[int] = None,
        num_samples: Optional[int] = None,
        score_type: str = "nll_like",
    ):
        """
        return:
            score_h: [B, pred_len]

        score_type:
            mse:
                样本均值预测误差

            min_mse:
                多样本中最接近真实 y 的误差

            mean_mse:
                多样本平均误差

            nll_like:
                简化概率式异常分数
        """

        samples = self.sample_heun(
            x_enc,
            num_steps=num_steps,
            num_samples=num_samples,
            return_all_samples=True,
        )

        # samples: [S, B, H]
        # y:       [B, H]

        if score_type == "mse":
            pred = samples.mean(dim=0)
            score_h = (pred - y) ** 2

        elif score_type == "min_mse":
            err = (samples - y.unsqueeze(0)) ** 2
            score_h = err.min(dim=0).values

        elif score_type == "mean_mse":
            err = (samples - y.unsqueeze(0)) ** 2
            score_h = err.mean(dim=0)

        elif score_type == "nll_like":
            mu = samples.mean(dim=0)
            var = samples.var(dim=0) + 1e-5
            score_h = ((y - mu) ** 2) / var + torch.log(var)

        else:
            raise ValueError(f"Unknown score_type: {score_type}")

        return score_h

    @torch.no_grad()
    def anomaly_score(
        self,
        x_enc,
        y,
        num_steps: Optional[int] = None,
        num_samples: Optional[int] = None,
        score_type: str = "nll_like",
    ):
        """
        return:
            score: [B]
        """
        score_h = self.anomaly_score_per_horizon(
            x_enc=x_enc,
            y=y,
            num_steps=num_steps,
            num_samples=num_samples,
            score_type=score_type,
        )

        return score_h.mean(dim=-1)
