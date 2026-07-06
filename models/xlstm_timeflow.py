import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

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


# ============================================================
# 1. Time embedding
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
    """
    t: [B] or [B, 1]
    return: [B, dim]
    """

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

        args = t[:, None] * freqs[None, :]  # [B, half_dim]

        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


# ============================================================
# 2. MLP block for vector field
# ============================================================

class ResidualMLPBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ============================================================
# 3. Conditional flow head
# ============================================================

class ConditionalFlowHead(nn.Module):
    """
    条件向量场 v_theta(y_t, t | context)

    y_t:     [B, pred_len]
    t:       [B]
    context: [B, d_model]

    output:
        velocity: [B, pred_len]
    """

    def __init__(
        self,
        pred_len: int = 4,
        d_model: int = 128,
        time_emb_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.pred_len = pred_len
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

    def forward(
        self,
        y_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        y_t: [B, pred_len]
        t: [B] or [B, 1]
        context: [B, d_model]
        """

        t_emb = self.time_emb(t)  # [B, time_emb_dim]

        h = torch.cat([y_t, t_emb, context], dim=-1)
        h = self.input_proj(h)

        for block in self.blocks:
            h = block(h)

        velocity = self.out_proj(self.out_norm(h))

        return velocity


# ============================================================
# 4. xLSTM + TimeFlow anomaly detector
# ============================================================

@dataclass
class xLSTMTimeFlowConfig:
    seq_len: int = 32
    pred_len: int = 4
    enc_in: int = 16

    d_model: int = 128
    num_blocks: int = 4
    num_heads: int = 4
    dropout: float = 0.1

    flow_hidden_dim: int = 256
    flow_layers: int = 3
    time_emb_dim: int = 64

    # sLSTM 放在哪些 block。
    # 如果 sLSTM CUDA kernel 编译麻烦，可以设置为空列表 []，只用 mLSTM。
    slstm_at: Optional[List[int]] = None

    # ODE sampling
    num_sampling_steps: int = 20
    num_samples: int = 1


class xLSTMTimeFlowAD(nn.Module):
    """
    xLSTM + conditional flow matching forecasting model.

    用于 UAV 目标飞参异常检测。

    输入:
        x: [B, 32, enc_in]
           除目标飞参外的其他飞参

    训练:
        y: [B, 4]
           目标飞参未来 4 个时间步

    推理:
        pred: [B, 4]
    """

    def __init__(self, config: xLSTMTimeFlowConfig):
        super().__init__()

        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.enc_in = config.enc_in
        self.d_model = config.d_model

        if config.slstm_at is None:
            # 默认只在第 1 层放一个 sLSTM，其余用 mLSTM
            # 如果你想更稳地跑通，可以传 slstm_at=[]
            slstm_at = [1] if config.num_blocks > 1 else [0]
        else:
            slstm_at = config.slstm_at

        self.input_proj = nn.Sequential(
            nn.Linear(config.enc_in, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
        )

        xlstm_cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=config.num_heads,
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda" if torch.cuda.is_available() else "vanilla",
                    num_heads=config.num_heads,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=1.3,
                    act_fn="gelu",
                ),
            ),
            context_length=config.seq_len,
            num_blocks=config.num_blocks,
            embedding_dim=config.d_model,
            slstm_at=slstm_at,
        )

        self.encoder = xLSTMBlockStack(xlstm_cfg)

        self.context_norm = nn.LayerNorm(config.d_model)

        self.flow_head = ConditionalFlowHead(
            pred_len=config.pred_len,
            d_model=config.d_model,
            time_emb_dim=config.time_emb_dim,
            hidden_dim=config.flow_hidden_dim,
            num_layers=config.flow_layers,
            dropout=config.dropout,
        )

        # 一个辅助的点预测头，可选。
        # 训练时可以不用；推理时可以作为 deterministic forecast。
        self.mean_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.pred_len),
        )

    # --------------------------------------------------------
    # Encode history
    # --------------------------------------------------------

    def encode_context(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, seq_len, enc_in]
        return:
            context: [B, d_model]
        """

        B, T, C = x.shape

        assert T == self.seq_len, f"Expected seq_len={self.seq_len}, got {T}"
        assert C == self.enc_in, f"Expected enc_in={self.enc_in}, got {C}"

        h = self.input_proj(x)      # [B, T, D]
        h = self.encoder(h)         # [B, T, D]

        # 用最后一个时间步作为历史状态表示
        context = self.context_norm(h[:, -1, :])  # [B, D]

        return context

    # --------------------------------------------------------
    # Conditional vector field
    # --------------------------------------------------------

    def vector_field(
        self,
        y_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        y_t: [B, pred_len]
        t: [B] or [B, 1]
        context: [B, d_model]
        return:
            velocity: [B, pred_len]
        """
        return self.flow_head(y_t, t, context)

    # --------------------------------------------------------
    # Flow matching training loss
    # --------------------------------------------------------

    def flow_matching_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Rectified Flow / Conditional Flow Matching loss.

        x: [B, 32, enc_in]
        y: [B, 4]

        采样:
            z0 ~ N(0, I)
            t ~ Uniform(0, 1)

        构造线性路径:
            y_t = (1 - t) * z0 + t * y

        目标速度:
            u_t = y - z0

        训练:
            MSE(v_theta(y_t, t | x), u_t)
        """

        B = y.size(0)
        device = y.device

        context = self.encode_context(x)  # [B, D]

        if noise is None:
            noise = torch.randn_like(y)   # z0: [B, 4]

        if t is None:
            t = torch.rand(B, device=device)  # [B]

        t_view = t.view(B, 1)

        y_t = (1.0 - t_view) * noise + t_view * y
        target_velocity = y - noise

        pred_velocity = self.vector_field(
            y_t=y_t,
            t=t,
            context=context,
        )

        loss = F.mse_loss(pred_velocity, target_velocity)

        return loss

    def mean_prediction_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        辅助 MSE loss，可选。
        让 context 本身也具备直接预测能力。
        """
        context = self.encode_context(x)
        pred = self.mean_head(context)
        return F.mse_loss(pred, y)

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lambda_mean: float = 0.0,
    ) -> torch.Tensor:
        """
        总损失。

        默认只用 flow matching loss。
        如果训练不稳定，可以设 lambda_mean=0.1。
        """

        loss_fm = self.flow_matching_loss(x, y)

        if lambda_mean > 0:
            loss_mean = self.mean_prediction_loss(x, y)
            return loss_fm + lambda_mean * loss_mean

        return loss_fm

    # --------------------------------------------------------
    # ODE sampling
    # --------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        num_steps: Optional[int] = None,
        num_samples: Optional[int] = None,
        return_all_samples: bool = False,
    ) -> torch.Tensor:
        """
        从噪声通过 ODE 生成未来目标飞参。

        x: [B, 32, enc_in]

        return:
            如果 return_all_samples=False:
                pred_mean: [B, 4]

            如果 return_all_samples=True:
                samples: [S, B, 4]
        """

        self.eval()

        if num_steps is None:
            num_steps = self.config.num_sampling_steps

        if num_samples is None:
            num_samples = self.config.num_samples

        B = x.size(0)
        device = x.device

        context = self.encode_context(x)  # [B, D]

        all_samples = []

        for _ in range(num_samples):
            y_t = torch.randn(
                B,
                self.pred_len,
                device=device,
                dtype=x.dtype,
            )

            # Euler ODE integration: t = 0 -> 1
            dt = 1.0 / num_steps

            for i in range(num_steps):
                t_value = torch.full(
                    (B,),
                    fill_value=i / num_steps,
                    device=device,
                    dtype=x.dtype,
                )

                velocity = self.vector_field(
                    y_t=y_t,
                    t=t_value,
                    context=context,
                )

                y_t = y_t + dt * velocity

            all_samples.append(y_t)

        samples = torch.stack(all_samples, dim=0)  # [S, B, 4]

        if return_all_samples:
            return samples

        return samples.mean(dim=0)  # [B, 4]

    @torch.no_grad()
    def sample_heun(
        self,
        x: torch.Tensor,
        num_steps: Optional[int] = None,
        num_samples: Optional[int] = None,
        return_all_samples: bool = False,
    ) -> torch.Tensor:
        """
        Heun / improved Euler sampling。
        比普通 Euler 稍慢，但通常更稳。
        """

        self.eval()

        if num_steps is None:
            num_steps = self.config.num_sampling_steps

        if num_samples is None:
            num_samples = self.config.num_samples

        B = x.size(0)
        device = x.device

        context = self.encode_context(x)

        all_samples = []

        for _ in range(num_samples):
            y_t = torch.randn(
                B,
                self.pred_len,
                device=device,
                dtype=x.dtype,
            )

            dt = 1.0 / num_steps

            for i in range(num_steps):
                t0 = torch.full(
                    (B,),
                    fill_value=i / num_steps,
                    device=device,
                    dtype=x.dtype,
                )
                t1 = torch.full(
                    (B,),
                    fill_value=(i + 1) / num_steps,
                    device=device,
                    dtype=x.dtype,
                )

                v0 = self.vector_field(y_t, t0, context)
                y_euler = y_t + dt * v0

                v1 = self.vector_field(y_euler, t1, context)

                y_t = y_t + 0.5 * dt * (v0 + v1)

            all_samples.append(y_t)

        samples = torch.stack(all_samples, dim=0)  # [S, B, 4]

        if return_all_samples:
            return samples

        return samples.mean(dim=0)

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        num_steps: Optional[int] = None,
        num_samples: Optional[int] = None,
        sampler: str = "heun",
    ) -> torch.Tensor:
        """
        推理接口。

        x: [B, 32, enc_in]
        return:
            pred: [B, 4]
        """

        if sampler == "euler":
            return self.sample(
                x,
                num_steps=num_steps,
                num_samples=num_samples,
                return_all_samples=False,
            )

        elif sampler == "heun":
            return self.sample_heun(
                x,
                num_steps=num_steps,
                num_samples=num_samples,
                return_all_samples=False,
            )

        elif sampler == "mean":
            context = self.encode_context(x)
            return self.mean_head(context)

        else:
            raise ValueError(f"Unknown sampler: {sampler}")

    # --------------------------------------------------------
    # Anomaly score
    # --------------------------------------------------------

    @torch.no_grad()
    def anomaly_score(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        num_steps: Optional[int] = None,
        num_samples: int = 8,
        score_type: str = "min_mse",
        sampler: str = "heun",
    ) -> torch.Tensor:
        """
        x: [B, 32, enc_in]
        y: [B, 4]

        return:
            score: [B]

        score_type:
            "mse":
                用 samples mean 和 y 的 MSE

            "min_mse":
                生成多个样本，取距离 y 最近的样本误差。
                适合概率预测，因为未来可能是多模态的。

            "mean_mse":
                所有样本误差取平均。
                更保守，异常分数可能更大。

            "nll_like":
                一个简化的概率式分数：
                mean_mse + log variance penalty
        """

        self.eval()

        if score_type == "mse":
            pred = self.forward(
                x,
                num_steps=num_steps,
                num_samples=1,
                sampler=sampler,
            )
            score = ((pred - y) ** 2).mean(dim=-1)
            return score

        if sampler == "euler":
            samples = self.sample(
                x,
                num_steps=num_steps,
                num_samples=num_samples,
                return_all_samples=True,
            )
        elif sampler == "heun":
            samples = self.sample_heun(
                x,
                num_steps=num_steps,
                num_samples=num_samples,
                return_all_samples=True,
            )
        else:
            raise ValueError("For multi-sample anomaly_score, sampler must be 'euler' or 'heun'.")

        # samples: [S, B, 4]
        # y:       [B, 4]
        err = (samples - y.unsqueeze(0)) ** 2    # [S, B, 4]
        mse_per_sample = err.mean(dim=-1)        # [S, B]

        if score_type == "min_mse":
            score = mse_per_sample.min(dim=0).values  # [B]

        elif score_type == "mean_mse":
            score = mse_per_sample.mean(dim=0)        # [B]

        elif score_type == "nll_like":
            mu = samples.mean(dim=0)                  # [B, 4]
            var = samples.var(dim=0) + 1e-5           # [B, 4]
            score = (((y - mu) ** 2) / var + torch.log(var)).mean(dim=-1)

        else:
            raise ValueError(f"Unknown score_type: {score_type}")

        return score
    