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


class PointUAVxLSTMAD(nn.Module):
    """
    不分 patch 的点级 xLSTM 预测式异常检测模型。

    Input:
        x: [B, T, C]
           B = batch size
           T = 输入窗口长度，例如 40 / 64 / 128 / 256
           C = UAV 多变量通道数

    Output:
        pred: [B, H, C]
              H = 预测步长，例如 1 / 4 / 8 / 16

    异常分数:
        score = mean((pred - y) ** 2, dim=-1)
        shape: [B, H]
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int = 128,
        pred_len: int = 1,
        d_model: int = 128,
        num_blocks: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        slstm_at=None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        if slstm_at is None:
            # 可以理解为：第 1 层用 sLSTM，其余层主要用 mLSTM。
            # 如果编译 sLSTM CUDA kernel 麻烦，可以先设成 []，只用 mLSTM。
            slstm_at = [1] if num_blocks > 1 else [0]

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=num_heads,
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda" if torch.cuda.is_available() else "vanilla",
                    num_heads=num_heads,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=1.3,
                    act_fn="gelu",
                ),
            ),
            context_length=seq_len,
            num_blocks=num_blocks,
            embedding_dim=d_model,
            slstm_at=slstm_at,
        )

        self.backbone = xLSTMBlockStack(cfg)

        self.norm = nn.LayerNorm(d_model)

        # 用最后一个时间步的 hidden state 预测未来 H 步
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len * input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C]
        return:
            pred: [B, H, C]
        """
        B, T, C = x.shape

        assert T == self.seq_len, f"Expected seq_len={self.seq_len}, got T={T}"
        assert C == self.input_dim, f"Expected input_dim={self.input_dim}, got C={C}"

        z = self.input_proj(x)          # [B, T, D]
        z = self.backbone(z)            # [B, T, D]
        z_last = self.norm(z[:, -1])    # [B, D]

        pred = self.pred_head(z_last)   # [B, H*C]
        pred = pred.view(B, self.pred_len, C)

        return pred

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C]
        y: [B, H, C]
        """
        pred = self.forward(x)
        return F.mse_loss(pred, y)

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        return:
            score: [B, H]
        """
        pred = self.forward(x)
        err = (pred - y) ** 2           # [B, H, C]
        score = err.mean(dim=-1)        # [B, H]
        return score