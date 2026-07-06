import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate

warnings.filterwarnings("ignore")


class Exp_xLSTMTimeFlow_AD(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=getattr(self.args, "weight_decay", 1e-4),
        )
        return optimizer

    def vali(self, vali_data, vali_loader):
        total_loss = []

        self.model.eval()

        with torch.no_grad():
            for batch_x, batch_y, batch_label, batch_index in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                loss = self.model.loss(batch_x, batch_y)

                total_loss.append(loss.item())

        total_loss = np.average(total_loss)

        self.model.train()

        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()

        train_steps = len(train_loader)

        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
        )

        model_optim = self._select_optimizer()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_label, batch_index) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)  # [B, 32, enc_in]
                batch_y = batch_y.float().to(self.device)  # [B, 4]

                loss = self.model.loss(batch_x, batch_y)

                train_loss.append(loss.item())

                loss.backward()

                if getattr(self.args, "grad_clip", 1.0) is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        getattr(self.args, "grad_clip", 1.0),
                    )

                model_optim.step()

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )

                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1,
                            epoch + 1,
                            loss.item(),
                        )
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed,
                            left_time,
                        )
                    )

                    iter_count = 0
                    time_now = time.time()

            print(
                "Epoch: {} cost time: {}".format(
                    epoch + 1,
                    time.time() - epoch_time,
                )
            )

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    vali_loss,
                )
            )

            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    @torch.no_grad()
    def _collect_point_scores(self, data_set, data_loader):
        """
        将 [B, 4] 的 horizon score 聚合成点级 score。

        return:
            point_scores: [total_len]
            point_labels: [total_len]
        """

        self.model.eval()

        total_len = data_set.total_len

        score_sum = np.zeros(total_len, dtype=np.float64)
        score_cnt = np.zeros(total_len, dtype=np.float64)

        point_labels = data_set.global_labels.copy()

        score_type = getattr(self.args, "score_type", "nll_like")
        num_steps = getattr(self.args, "num_sampling_steps", 20)
        num_samples = getattr(self.args, "num_samples", 8)

        for batch_x, batch_y, batch_label, batch_index in data_loader:
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            score_h = self.model.anomaly_score_per_horizon(
                x_enc=batch_x,
                y=batch_y,
                num_steps=num_steps,
                num_samples=num_samples,
                score_type=score_type,
            )

            score_h = score_h.detach().cpu().numpy()      # [B, 4]
            batch_index = batch_index.detach().cpu().numpy()  # [B, 4]

            B, H = score_h.shape

            for b in range(B):
                for h in range(H):
                    idx = int(batch_index[b, h])
                    score_sum[idx] += float(score_h[b, h])
                    score_cnt[idx] += 1.0

        point_scores = score_sum / np.maximum(score_cnt, 1.0)
        point_scores[score_cnt == 0] = np.nan

        return point_scores, point_labels

    def _fit_threshold(self, train_scores):
        valid = ~np.isnan(train_scores)
        s = train_scores[valid]

        threshold_type = getattr(self.args, "threshold_type", "3sigma")

        if threshold_type == "3sigma":
            mu = s.mean()
            sigma = s.std()
            threshold = mu + 3.0 * sigma

        elif threshold_type == "quantile":
            q = getattr(self.args, "threshold_q", 0.995)
            threshold = np.quantile(s, q)

        else:
            raise ValueError(f"Unknown threshold_type: {threshold_type}")

        return float(threshold)

    def test(self, setting, test=0):
        train_data, train_loader = self._get_data(flag="train")
        test_data, test_loader = self._get_data(flag="test")

        if test:
            print("loading model")
            ckpt_path = os.path.join(
                self.args.checkpoints,
                setting,
                "checkpoint.pth",
            )
            self.model.load_state_dict(torch.load(ckpt_path))

        print("Collect train scores for threshold...")
        train_scores, _ = self._collect_point_scores(train_data, train_loader)

        threshold = self._fit_threshold(train_scores)

        print(f"Threshold = {threshold:.6f}")

        print("Collect test scores...")
        test_scores, test_labels = self._collect_point_scores(test_data, test_loader)

        valid = ~np.isnan(test_scores)

        test_scores = test_scores[valid]
        test_labels = test_labels[valid]

        test_pred = (test_scores > threshold).astype(np.int64)
        test_true = test_labels.astype(np.int64)

        accuracy = accuracy_score(test_true, test_pred)

        precision, recall, f_score, support = precision_recall_fscore_support(
            test_true,
            test_pred,
            average="binary",
            pos_label=1,
            zero_division=0,
        )

        print("Anomaly Detection Result")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F-score  : {f_score:.4f}")

        folder_path = os.path.join("./test_results", setting)
        os.makedirs(folder_path, exist_ok=True)

        np.save(os.path.join(folder_path, "test_scores.npy"), test_scores)
        np.save(os.path.join(folder_path, "test_labels.npy"), test_true)
        np.save(os.path.join(folder_path, "test_pred.npy"), test_pred)
        np.save(os.path.join(folder_path, "threshold.npy"), np.array([threshold]))

        return
