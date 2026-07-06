import torch
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

# 1. 配置模型
cfg = xLSTMBlockStackConfig(
    # mLSTM块配置（矩阵记忆，长序列记忆强，可并行）
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4,  # 因果卷积核大小
            qkv_proj_blocksize=4,
            num_heads=4            # 注意力头数
        )
    ),
    # sLSTM块配置（标量记忆，状态跟踪能力强）
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            backend="cuda",       # 计算后端：cuda / native
            num_heads=4,
            conv1d_kernel_size=4,
            bias_init="powerlaw_blockdependent",
        ),
        feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
    ),
    context_length=256,    # 最大序列长度
    num_blocks=7,          # 总块数（层数）
    embedding_dim=128,     # 特征维度
    slstm_at=[1],          # 指定第1层用sLSTM，其余用mLSTM
)

# 2. 初始化模型
model = xLSTMBlockStack(cfg).to("cuda")

# 3. 前向传播
x = torch.randn(4, 256, 128).to("cuda")  # [batch, seq_len, dim]
output = model(x)
print(output.shape)  # torch.Size([4, 256, 128])