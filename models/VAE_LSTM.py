import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    VAE-LSTM for TSLib anomaly_detection.

    Input:  x_enc [B, T, C]
    Output: recon [B, T, C]

    Paper setting:
      seq_len=48, enc_in=3, latent_dim=10, lstm_hidden=64
      Use reconstruction error for anomaly score.
    """

    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.latent_dim = getattr(configs, "latent_dim", 10)
        self.lstm_hidden = getattr(configs, "lstm_hidden", 64)
        self.beta_kl = getattr(configs, "beta_kl", 1e-3)

        if self.seq_len % 8 != 0:
            raise ValueError("VAE_LSTM expects seq_len divisible by 8, e.g. 48, 96, 192.")

        h4 = self.seq_len // 8

        # x: [B, 1, T, C]
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(2, 1), padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1),
            nn.LeakyReLU(inplace=True),
            # Collapse time and variable dimensions, same idea as paper's Conv2d_4: kernel=(6,3)
            nn.Conv2d(128, 512, kernel_size=(h4, self.enc_in), stride=1, padding=0),
            nn.LeakyReLU(inplace=True),
        )

        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.LeakyReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(512, self.latent_dim)
        self.fc_logvar = nn.Linear(512, self.latent_dim)

        self.lstm = nn.LSTM(
            input_size=self.latent_dim,
            hidden_size=self.lstm_hidden,
            num_layers=2,
            batch_first=True,
        )
        self.lstm_out = nn.Linear(self.lstm_hidden, self.latent_dim)

        self.decoder_fc = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.LeakyReLU(inplace=True),
        )

        # z -> [B,512,1,1] -> [B,1,T,C]
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=(h4, self.enc_in), stride=1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),
        )

        self.last_mu = None
        self.last_logvar = None

    def encode(self, x):
        # x: [B,T,C]
        h = self.encoder_cnn(x.unsqueeze(1))
        h = self.encoder_fc(h)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z).view(z.size(0), 512, 1, 1)
        out = self.decoder_deconv(h).squeeze(1)  # [B,T,C]
        return out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        mu, logvar = self.encode(x_enc)
        self.last_mu = mu
        self.last_logvar = logvar

        z = self.reparameterize(mu, logvar)

        # TSLib default sample is one window. Treat its latent as one-step embedding sequence.
        z_seq = z.unsqueeze(1)              # [B,1,Z]
        z_lstm, _ = self.lstm(z_seq)
        z_hat = self.lstm_out(z_lstm[:, -1])

        recon = self.decode(z_hat)

        # Safety crop/pad in case custom seq_len causes one-off size mismatch.
        recon = recon[:, :self.seq_len, :self.enc_in]
        if recon.shape[1] != self.seq_len or recon.shape[2] != self.enc_in:
            recon = F.interpolate(
                recon.permute(0, 2, 1),
                size=self.seq_len,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)
            recon = recon[:, :, :self.enc_in]
        return recon

    def vae_loss(self, recon, target):
        rec = F.mse_loss(recon, target)
        if self.last_mu is None or self.last_logvar is None:
            return rec
        kl = -0.5 * torch.mean(1 + self.last_logvar - self.last_mu.pow(2) - self.last_logvar.exp())
        return rec + self.beta_kl * kl

