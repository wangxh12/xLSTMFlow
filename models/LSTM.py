import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Predictive LSTM baseline for anomaly_detection_pred.

    当前任务:
        输入:
            x_enc: [B, 40, 11]
            对应字段:
            ["alt", "h", "navvn", "navve", "navvd",
             "vn", "ve", "vd", "p", "q", "r"]

        输出:
            pred: [B, 4, 1]
            对应预测目标:
            ["navalt"]

    训练方式:
        Exp_Anomaly_Detection_Pred 中会自动使用:
            loss = MSE(pred, batch_y)

    异常分数:
        仍然使用你当前 exp_anomaly_detection_pred.py 里的:
            residual = |pred - true|
            或者你当前配置中的残差分数。
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
        self.e_layers = configs.e_layers
        self.dropout = configs.dropout

        self.input_projection = nn.Linear(self.enc_in, self.d_model)

        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.e_layers,
            dropout=self.dropout if self.e_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False
        )

        self.norm = nn.LayerNorm(self.d_model)

        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len * self.c_out)
        )

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        x_enc:
            [B, seq_len, enc_in]

        return:
            [B, pred_len, c_out]
        """
        B, L, C = x_enc.shape

        if C != self.enc_in:
            raise RuntimeError(
                f"Input channel mismatch: got {C}, expected {self.enc_in}"
            )

        x = self.input_projection(x_enc)
        # [B, L, d_model]

        out, (h_n, c_n) = self.lstm(x)
        # out: [B, L, d_model]

        # 使用最后一个时间步的 hidden state 作为历史窗口表征
        h = out[:, -1, :]
        h = self.norm(h)

        pred = self.head(h)
        pred = pred.reshape(B, self.pred_len, self.c_out)

        return pred

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in [
            "long_term_forecast",
            "short_term_forecast",
            "anomaly_detection_pred"
        ]:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if self.task_name == "anomaly_detection":
            raise RuntimeError(
                "LSTMBaseline is a predictive anomaly detection model. "
                "It outputs [B, pred_len, c_out], but the original TSLib "
                "anomaly_detection Exp expects reconstruction output [B, seq_len, C]. "
                "Please use task_name='anomaly_detection_pred'."
            )

        raise NotImplementedError(
            f"LSTMBaseline does not support task_name={self.task_name}"
        )
