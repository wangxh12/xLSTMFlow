import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class VariableAttention(nn.Module):
    """
    Variable-wise attention.

    Input:
        x: [B, N, C, D]
           B: batch size
           N: number of temporal patches
           C: number of input variables
           D: hidden dimension

    For each temporal patch, this module performs attention across variables.
    """
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C, D = x.shape

        x_reshape = x.reshape(B * N, C, D)

        attn_out, _ = self.attn(
            x_reshape,
            x_reshape,
            x_reshape,
            need_weights=False
        )

        x_reshape = self.norm(x_reshape + self.dropout(attn_out))

        return x_reshape.reshape(B, N, C, D)


class SimplexLSTMBlock(nn.Module):
    """
    Lightweight xLSTM-like recurrent block.

    This is not the official xLSTM implementation.
    It keeps the useful design idea for this prototype:
        1. exponential-style input/forget gates
        2. normalized gate competition
        3. recurrent temporal modeling
        4. residual connection
        5. RMSNorm
        6. GLU feed-forward layer

    Input:
        x: [B, N, D]

    Output:
        x: [B, N, D]
    """
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()

        self.in_proj = nn.Linear(d_model, d_model * 4)
        self.h_proj = nn.Linear(d_model, d_model * 4)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape

        h = x.new_zeros(B, D)
        c = x.new_zeros(B, D)

        outputs = []

        for t in range(N):
            xt = x[:, t, :]

            gates = self.in_proj(xt) + self.h_proj(h)
            i, f, o, g = gates.chunk(4, dim=-1)

            # Exponential-style gates.
            # Clamp is necessary; otherwise exp gate can easily explode.
            i = torch.exp(torch.clamp(i, max=5.0))
            f = torch.exp(torch.clamp(f, max=5.0))

            # Normalize input and forget gates.
            gate_sum = i + f + 1e-6
            i = i / gate_sum
            f = f / gate_sum

            o = torch.sigmoid(o)
            g = torch.tanh(g)

            c = f * c + i * g
            h = o * torch.tanh(c)

            outputs.append(h)

        y = torch.stack(outputs, dim=1)
        y = self.out_proj(y)

        x = self.norm1(x + self.dropout(y))
        x = self.norm2(x + self.dropout(self.ffn(x)))

        return x


class FlowVectorField(nn.Module):
    """
    Conditional Flow Matching vector field.

    It predicts:
        v_theta(z_t, t, condition)

    z_t:
        [B, pred_dim]

    t:
        [B, 1]

    condition:
        [B, D]

    output:
        [B, pred_dim]
    """
    def __init__(self, pred_dim, d_model, dropout):
        super().__init__()

        self.z_proj = nn.Linear(pred_dim, d_model)

        self.t_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.cond_proj = nn.Linear(d_model, d_model)

        self.net = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, pred_dim),
        )

    def forward(self, z_t, t, condition):
        z_emb = self.z_proj(z_t)
        t_emb = self.t_proj(t)
        c_emb = self.cond_proj(condition)

        h = torch.cat([z_emb, t_emb, c_emb], dim=-1)

        return self.net(h)


class Model(nn.Module):
    """
    xLSTMFlow.

    Predictive model:

        past multivariate window
                ->
        future target patch

    For your current setting:

        x_enc:
            [B, 40, 11]

        output:
            [B, 4, 1]

    Recommended args:

        --model xLSTMFlow
        --seq_len 40
        --pred_len 4
        --patch_len 4
        --enc_in 11
        --c_out 1
        --d_model 128
        --d_ff 256
        --e_layers 2
        --n_heads 4

    Notes:
        1. This file is compatible with the latest TSLib model loading style:
           put this file under ./models and use --model xLSTMFlow.
        2. For true Flow Matching training, call model.flow_matching_loss(x_enc, y_true)
           in your predictive anomaly detection Exp.
        3. For ordinary forecasting Exp, forward() returns deterministic ODE prediction
           and can be trained with MSE, but that is not the full Flow Matching objective.
    """

    def __init__(self, configs):
        super().__init__()

        self.configs = configs
        self.task_name = configs.task_name

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.e_layers = configs.e_layers
        self.n_heads = configs.n_heads
        self.dropout = configs.dropout

        self.patch_len = getattr(configs, "patch_len", 4)
        self.stride = getattr(configs, "stride", self.patch_len)
        self.fm_steps = getattr(configs, "fm_steps", 8)

        assert self.seq_len >= self.patch_len, "seq_len must be >= patch_len"
        assert self.pred_len > 0, "pred_len must be > 0"
        assert self.enc_in > 0, "enc_in must be > 0"
        assert self.c_out > 0, "c_out must be > 0"

        self.patch_num = (self.seq_len - self.patch_len) // self.stride + 1
        self.pred_dim = self.pred_len * self.c_out

        self.patch_proj = nn.Linear(self.patch_len, self.d_model)

        self.channel_embedding = nn.Parameter(
            torch.randn(1, 1, self.enc_in, self.d_model) * 0.02
        )

        self.position_embedding = nn.Parameter(
            torch.randn(1, self.patch_num, 1, self.d_model) * 0.02
        )

        self.spatial_mixer = VariableAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout=self.dropout
        )

        self.temporal_blocks = nn.ModuleList([
            SimplexLSTMBlock(
                d_model=self.d_model,
                d_ff=self.d_ff,
                dropout=self.dropout
            )
            for _ in range(self.e_layers)
        ])

        self.final_norm = RMSNorm(self.d_model)

        self.flow = FlowVectorField(
            pred_dim=self.pred_dim,
            d_model=self.d_model,
            dropout=self.dropout
        )

    def _patchify(self, x_enc):
        """
        x_enc:
            [B, L, C]

        return:
            [B, patch_num, C, patch_len]
        """
        B, L, C = x_enc.shape

        assert C == self.enc_in, \
            f"Input channel mismatch: got {C}, expected {self.enc_in}"

        patches = x_enc.unfold(
            dimension=1,
            size=self.patch_len,
            step=self.stride
        )

        # PyTorch unfold result:
        # [B, patch_num, C, patch_len]
        return patches.contiguous()

    def encode(self, x_enc):
        """
        Encode historical multivariate window.

        x_enc:
            [B, seq_len, enc_in]

        condition:
            [B, d_model]
        """
        patches = self._patchify(x_enc)
        B, N, C, P = patches.shape

        x = self.patch_proj(patches)
        # [B, N, C, D]

        x = x + self.channel_embedding[:, :, :C, :]
        x = x + self.position_embedding[:, :N, :, :]

        # spatial variable interaction
        x = self.spatial_mixer(x)
        # [B, N, C, D]

        # merge variable tokens into one temporal token per patch
        x = x.mean(dim=2)
        # [B, N, D]

        # temporal modeling over patches
        for block in self.temporal_blocks:
            x = block(x)

        x = self.final_norm(x)

        # use the last patch token as condition
        condition = x[:, -1, :]

        return condition

    def flow_matching_loss(self, x_enc, y_true):
        """
        Conditional Flow Matching loss.

        x_enc:
            [B, seq_len, enc_in]

        y_true:
            [B, pred_len, c_out]

        For your case:
            x_enc:  [B, 40, 11]
            y_true: [B, 4, 1]
        """
        B = x_enc.shape[0]

        assert y_true.shape[1] == self.pred_len, \
            f"y_true pred_len mismatch: got {y_true.shape[1]}, expected {self.pred_len}"

        assert y_true.shape[2] == self.c_out, \
            f"y_true channel mismatch: got {y_true.shape[2]}, expected {self.c_out}"

        condition = self.encode(x_enc)

        x1 = y_true.reshape(B, self.pred_dim)
        x0 = torch.randn_like(x1)

        t = torch.rand(B, 1, device=x_enc.device)

        z_t = (1.0 - t) * x0 + t * x1
        target_velocity = x1 - x0

        pred_velocity = self.flow(z_t, t, condition)

        loss = F.mse_loss(pred_velocity, target_velocity)

        return loss

    def sample(self, x_enc, steps=None, stochastic=False):
        """
        Generate future target patch through Euler ODE integration.

        x_enc:
            [B, seq_len, enc_in]

        output:
            [B, pred_len, c_out]

        stochastic=False:
            start from zeros. More stable for deterministic anomaly scoring.

        stochastic=True:
            start from Gaussian noise. Useful for multi-sample uncertainty scoring.
        """
        steps = steps or self.fm_steps

        B = x_enc.shape[0]
        condition = self.encode(x_enc)

        if stochastic:
            z = torch.randn(B, self.pred_dim, device=x_enc.device)
        else:
            z = torch.zeros(B, self.pred_dim, device=x_enc.device)

        dt = 1.0 / float(steps)

        for i in range(steps):
            t_value = torch.full(
                (B, 1),
                fill_value=float(i) / float(steps),
                device=x_enc.device
            )

            velocity = self.flow(z, t_value, condition)
            z = z + velocity * dt

        pred = z.reshape(B, self.pred_len, self.c_out)

        return pred

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        Forecast future target patch.

        Return:
            [B, pred_len, c_out]
        """
        return self.sample(
            x_enc=x_enc,
            steps=self.fm_steps,
            stochastic=False
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        TSLib-style forward function.

        For predictive anomaly detection, use:
            task_name = anomaly_detection_pred

        For forecasting-style running, use:
            task_name = long_term_forecast or short_term_forecast

        Do not use the original TSLib anomaly_detection Exp directly,
        because the original Exp expects reconstruction output [B, seq_len, C].
        This model outputs prediction [B, pred_len, c_out].
        """
        if self.task_name in [
            "long_term_forecast",
            "short_term_forecast",
            "anomaly_detection_pred"
        ]:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if self.task_name == "anomaly_detection":
            raise RuntimeError(
                "xLSTMFlow is a predictive anomaly detection model. "
                "The original TSLib anomaly_detection Exp is reconstruction-based "
                "and expects output shape [B, seq_len, C]. "
                "Please use your predictive Exp, e.g. task_name='anomaly_detection_pred', "
                "or run it as a forecasting model."
            )

        raise NotImplementedError(
            f"xLSTMFlow does not support task_name={self.task_name}"
        )