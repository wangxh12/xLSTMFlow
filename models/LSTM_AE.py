import torch
import torch.nn as nn


class Model(nn.Module):
    """
    LSTM AutoEncoder for reconstruction-based anomaly detection.

    Input:
        x_enc: [B, T, C]

    Output:
        recon: [B, T, C]

    Recommended args:
        --model LSTM_AE
        --seq_len 48
        --enc_in 18
        --c_out 18
        --d_model 128
        --e_layers 2
        --d_layers 1
        --dropout 0.1
    """

    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.c_out = getattr(configs, "c_out", configs.enc_in)

        self.hidden_size = getattr(configs, "d_model", 128)
        self.latent_dim = getattr(configs, "latent_dim", self.hidden_size)

        self.encoder_layers = getattr(configs, "e_layers", 2)
        self.decoder_layers = getattr(configs, "d_layers", 1)
        self.dropout = getattr(configs, "dropout", 0.1)

        enc_dropout = self.dropout if self.encoder_layers > 1 else 0.0
        dec_dropout = self.dropout if self.decoder_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.encoder_layers,
            batch_first=True,
            dropout=enc_dropout,
            bidirectional=False,
        )

        self.to_latent = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.latent_dim),
            nn.Tanh(),
        )

        self.latent_to_dec = nn.Linear(self.latent_dim, self.hidden_size)

        self.decoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.decoder_layers,
            batch_first=True,
            dropout=dec_dropout,
            bidirectional=False,
        )

        self.projection = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.c_out),
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        x_enc: [B, T, C]
        """
        batch_size, seq_len, _ = x_enc.shape

        _, (h_n, _) = self.encoder(x_enc)

        # Last layer hidden state as sequence representation
        h_last = h_n[-1]                       # [B, hidden]
        z = self.to_latent(h_last)             # [B, latent_dim]

        # Repeat latent representation for each time step
        dec_input = self.latent_to_dec(z)      # [B, hidden]
        dec_input = dec_input.unsqueeze(1).repeat(1, seq_len, 1)

        dec_out, _ = self.decoder(dec_input)   # [B, T, hidden]
        recon = self.projection(dec_out)       # [B, T, c_out]

        return recon
