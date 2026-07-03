from matplotlib import pyplot as plt

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch import optim

import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection_Pred(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection_Pred, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        return nn.MSELoss()

    def _unwrap_model(self):
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    def _train_loss(self, batch_x, batch_y, criterion):
        model = self._unwrap_model()

        if hasattr(model, "flow_matching_loss"):
            return model.flow_matching_loss(batch_x, batch_y)

        outputs = self.model(batch_x, None, None, None)
        return criterion(outputs, batch_y)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_label) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)
                loss = criterion(outputs, batch_y)

                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_label) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                loss = self._train_loss(batch_x, batch_y, criterion)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, loss.item()
                    ))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )

            early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def _window_score(self, batch_x, batch_y):
        """
        返回:
            score: [B, pred_len]
        """
        outputs = self.model(batch_x, None, None, None)
        error = (outputs - batch_y) ** 2
        score = torch.mean(error, dim=-1)
        return score

    def _aggregate_scores(self, scores, labels, raw_len):
        """
        scores: [N, pred_len]
        labels: [N, pred_len]

        把预测窗口分数聚合回 point-level 分数。
        """
        pred_len = self.args.pred_len
        seq_len = self.args.seq_len

        point_score = np.zeros(raw_len, dtype=np.float32)
        point_count = np.zeros(raw_len, dtype=np.float32)
        point_label = np.zeros(raw_len, dtype=np.float32)

        for i in range(scores.shape[0]):
            start = i + seq_len
            end = start + pred_len

            if end > raw_len:
                end = raw_len

            valid_len = end - start
            if valid_len <= 0:
                continue

            point_score[start:end] += scores[i, :valid_len]
            point_count[start:end] += 1
            point_label[start:end] = np.maximum(point_label[start:end], labels[i, :valid_len])

        valid = point_count > 0

        point_score[valid] = point_score[valid] / point_count[valid]

        return point_score[valid], point_label[valid]

    def _window_score(self, batch_x, batch_y):
        """
        计算预测残差分数。

        batch_x:
            [B, seq_len, enc_in]

        batch_y:
            [B, pred_len, c_out]

        return:
            score: [B, pred_len]
        """
        outputs = self.model(batch_x, None, None, None)

        # 对 navalt 预测任务来说，c_out=1
        # residual: [B, pred_len, 1]
        residual = torch.abs(outputs - batch_y)

        # score: [B, pred_len]
        score = torch.mean(residual, dim=-1)

        return score

    def _aggregate_scores(self, scores, labels, raw_len):
        """
        将窗口级 future patch 分数聚合回 point-level 分数。

        scores:
            [N, pred_len]

        labels:
            [N, pred_len]

        raw_len:
            原始测试序列长度

        return:
            point_score_valid: [valid_len]
            point_label_valid: [valid_len]
        """
        pred_len = self.args.pred_len
        seq_len = self.args.seq_len

        point_score = np.zeros(raw_len, dtype=np.float32)
        point_count = np.zeros(raw_len, dtype=np.float32)
        point_label = np.zeros(raw_len, dtype=np.float32)

        for i in range(scores.shape[0]):
            start = i + seq_len
            end = start + pred_len

            if start >= raw_len:
                continue

            if end > raw_len:
                end = raw_len

            valid_len = end - start
            if valid_len <= 0:
                continue

            point_score[start:end] += scores[i, :valid_len]
            point_count[start:end] += 1.0

            # 一个点只要被任一窗口标为异常，就认为该点异常
            point_label[start:end] = np.maximum(
                point_label[start:end],
                labels[i, :valid_len]
            )

        valid = point_count > 0

        point_score[valid] = point_score[valid] / point_count[valid]

        return point_score[valid], point_label[valid]

    def _find_segments(self, labels):
        """
        找出连续异常区间。

        labels:
            1D array, 0/1

        return:
            [(start, end), ...]
            start/end 均为闭区间索引
        """
        labels = np.asarray(labels).astype(int).reshape(-1)

        segments = []
        in_segment = False
        start = 0

        for i, value in enumerate(labels):
            if value == 1 and not in_segment:
                start = i
                in_segment = True
            elif value == 0 and in_segment:
                segments.append((start, i - 1))
                in_segment = False

        if in_segment:
            segments.append((start, len(labels) - 1))

        return segments

    def _plot_residual_with_threshold(self, residual, threshold, gt, save_path, title):
        """
        画残差 + 阈值 + 真实异常背景。

        residual:
            [T]

        threshold:
            float

        gt:
            [T], 0/1
        """
        residual = np.asarray(residual).reshape(-1)
        gt = np.asarray(gt).astype(int).reshape(-1)

        assert len(residual) == len(gt), \
            f"residual length {len(residual)} != gt length {len(gt)}"

        x = np.arange(len(residual))

        plt.figure(figsize=(16, 5))
        ax = plt.gca()

        anomaly_segments = self._find_segments(gt)

        for idx, (start, end) in enumerate(anomaly_segments):
            ax.axvspan(
                start,
                end,
                color="lightcoral",
                alpha=0.25,
                label="Ground Truth Anomaly" if idx == 0 else None
            )

        ax.plot(
            x,
            residual,
            linewidth=1.2,
            label="Residual Score"
        )

        ax.axhline(
            y=threshold,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold = {threshold:.6f}"
        )

        ax.set_title(title)
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Residual")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    def _plot_prediction_result(self, residual, threshold, gt, pred, save_path, title):
        """
        画残差 + 阈值 + 真实异常背景 + 预测异常点。

        residual:
            [T]

        threshold:
            float

        gt:
            [T], 0/1

        pred:
            [T], 0/1
        """
        residual = np.asarray(residual).reshape(-1)
        gt = np.asarray(gt).astype(int).reshape(-1)
        pred = np.asarray(pred).astype(int).reshape(-1)

        assert len(residual) == len(gt), \
            f"residual length {len(residual)} != gt length {len(gt)}"

        assert len(residual) == len(pred), \
            f"residual length {len(residual)} != pred length {len(pred)}"

        x = np.arange(len(residual))

        plt.figure(figsize=(16, 5))
        ax = plt.gca()

        anomaly_segments = self._find_segments(gt)

        for idx, (start, end) in enumerate(anomaly_segments):
            ax.axvspan(
                start,
                end,
                color="lightcoral",
                alpha=0.25,
                label="Ground Truth Anomaly" if idx == 0 else None
            )

        ax.plot(
            x,
            residual,
            linewidth=1.2,
            label="Residual Score"
        )

        ax.axhline(
            y=threshold,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold = {threshold:.6f}"
        )

        pred_idx = np.where(pred == 1)[0]
        if len(pred_idx) > 0:
            ax.scatter(
                pred_idx,
                residual[pred_idx],
                s=12,
                alpha=0.8,
                label="Predicted Anomaly"
            )

        ax.set_title(title)
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Residual")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')

        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(
                    os.path.join('./checkpoints/' + setting, 'checkpoint.pth'),
                    map_location=self.device
                )
            )

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()

        # ==========================================================
        # 1. 计算训练集残差分数，用于确定阈值
        # ==========================================================
        train_scores = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_label) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                score = self._window_score(batch_x, batch_y)
                score = score.detach().cpu().numpy()

                train_scores.append(score)

        train_scores = np.concatenate(train_scores, axis=0)
        train_energy = train_scores.reshape(-1)

        # ==========================================================
        # 2. 计算测试集残差分数
        # ==========================================================
        test_scores = []
        test_labels = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_label) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                score = self._window_score(batch_x, batch_y)
                score = score.detach().cpu().numpy()

                test_scores.append(score)
                test_labels.append(batch_label.numpy())

        test_scores = np.concatenate(test_scores, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        raw_test_len = len(test_data.test)

        test_energy, gt = self._aggregate_scores(
            scores=test_scores,
            labels=test_labels,
            raw_len=raw_test_len
        )

        gt = gt.astype(int)

        # ==========================================================
        # 3. 计算阈值
        # ==========================================================
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        threshold = np.percentile(
            combined_energy,
            100 - self.args.anomaly_ratio
        )

        print("Threshold :", threshold)

        # ==========================================================
        # 4. 原始预测结果，不使用 point adjustment
        # ==========================================================
        pred_raw = (test_energy > threshold).astype(int)
        gt_raw = gt.astype(int)

        print("raw pred:", pred_raw.shape)
        print("raw gt:", gt_raw.shape)

        accuracy_raw = accuracy_score(gt_raw, pred_raw)

        precision_raw, recall_raw, f_score_raw, support_raw = precision_recall_fscore_support(
            gt_raw,
            pred_raw,
            average='binary',
            zero_division=0
        )

        print("Without adjustment:")
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(
                accuracy_raw,
                precision_raw,
                recall_raw,
                f_score_raw
            )
        )

        # ==========================================================
        # 5. 使用 TSLib point adjustment
        # ==========================================================
        gt_adj, pred_adj = adjustment(gt_raw.copy(), pred_raw.copy())

        gt_adj = np.asarray(gt_adj).astype(int)
        pred_adj = np.asarray(pred_adj).astype(int)

        print("adjusted pred:", pred_adj.shape)
        print("adjusted gt:", gt_adj.shape)

        accuracy_adj = accuracy_score(gt_adj, pred_adj)

        precision_adj, recall_adj, f_score_adj, support_adj = precision_recall_fscore_support(
            gt_adj,
            pred_adj,
            average='binary',
            zero_division=0
        )

        print("With adjustment:")
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(
                accuracy_adj,
                precision_adj,
                recall_adj,
                f_score_adj
            )
        )

        # ==========================================================
        # 6. 可视化：残差、阈值、异常背景
        # ==========================================================
        self._plot_residual_with_threshold(
            residual=test_energy,
            threshold=threshold,
            gt=gt_raw,
            save_path=os.path.join(folder_path, "residual_threshold.png"),
            title=f"{setting} | Residual Score and Threshold"
        )

        self._plot_prediction_result(
            residual=test_energy,
            threshold=threshold,
            gt=gt_raw,
            pred=pred_raw,
            save_path=os.path.join(folder_path, "residual_threshold_pred_raw.png"),
            title=f"{setting} | Raw Prediction"
        )

        self._plot_prediction_result(
            residual=test_energy,
            threshold=threshold,
            gt=gt_raw,
            pred=pred_adj,
            save_path=os.path.join(folder_path, "residual_threshold_pred_adjusted.png"),
            title=f"{setting} | Point-Adjusted Prediction"
        )

        # ==========================================================
        # 7. 保存结果
        # ==========================================================
        result_file = "result_anomaly_detection_pred.txt"

        with open(result_file, 'a') as f:
            f.write(setting + "\n")

            f.write("Without adjustment:\n")
            f.write(
                "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}\n".format(
                    accuracy_raw,
                    precision_raw,
                    recall_raw,
                    f_score_raw
                )
            )

            f.write("With adjustment:\n")
            f.write(
                "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}\n".format(
                    accuracy_adj,
                    precision_adj,
                    recall_adj,
                    f_score_adj
                )
            )

            f.write("\n")

        np.save(os.path.join(folder_path, "test_energy.npy"), test_energy)
        np.save(os.path.join(folder_path, "gt_raw.npy"), gt_raw)
        np.save(os.path.join(folder_path, "pred_raw.npy"), pred_raw)
        np.save(os.path.join(folder_path, "gt_adjusted.npy"), gt_adj)
        np.save(os.path.join(folder_path, "pred_adjusted.npy"), pred_adj)
        np.save(os.path.join(folder_path, "threshold.npy"), np.array([threshold]))

        print("Saved visualization to:", folder_path)
        print("Saved residual_threshold.png")
        print("Saved residual_threshold_pred_raw.png")
        print("Saved residual_threshold_pred_adjusted.png")

        return

    # def test(self, setting, test=0):
    #     test_data, test_loader = self._get_data(flag='test')
    #     train_data, train_loader = self._get_data(flag='train')

    #     if test:
    #         print('loading model')
    #         self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

    #     folder_path = './test_results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     self.model.eval()

    #     # 1. train energy
    #     train_scores = []

    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_label) in enumerate(train_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float().to(self.device)

    #             score = self._window_score(batch_x, batch_y)
    #             score = score.detach().cpu().numpy()

    #             train_scores.append(score)

    #     train_scores = np.concatenate(train_scores, axis=0)
    #     train_energy = train_scores.reshape(-1)

    #     # 2. test energy
    #     test_scores = []
    #     test_labels = []

    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_label) in enumerate(test_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float().to(self.device)

    #             score = self._window_score(batch_x, batch_y)
    #             score = score.detach().cpu().numpy()

    #             test_scores.append(score)
    #             test_labels.append(batch_label.numpy())

    #     test_scores = np.concatenate(test_scores, axis=0)
    #     test_labels = np.concatenate(test_labels, axis=0)

    #     raw_test_len = len(test_data.test)
    #     test_energy, gt = self._aggregate_scores(test_scores, test_labels, raw_test_len)

    #     # 3. threshold
    #     combined_energy = np.concatenate([train_energy, test_energy], axis=0)
    #     threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)

    #     print("Threshold :", threshold)

    #     # 4. evaluation
    #     pred = (test_energy > threshold).astype(int)
    #     gt = gt.astype(int)

    #     print("pred: ", pred.shape)
    #     print("gt: ", gt.shape)

    #     gt, pred = adjustment(gt, pred)

    #     pred = np.array(pred)
    #     gt = np.array(gt)

    #     print("pred: ", pred.shape)
    #     print("gt: ", gt.shape)

    #     accuracy = accuracy_score(gt, pred)
    #     precision, recall, f_score, support = precision_recall_fscore_support(
    #         gt, pred, average='binary'
    #     )

    #     print(
    #         "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
    #             accuracy, precision, recall, f_score
    #         )
    #     )

    #     f = open("result_anomaly_detection_pred.txt", 'a')
    #     f.write(setting + " \n")
    #     f.write(
    #         "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
    #             accuracy, precision, recall, f_score
    #         )
    #     )
    #     f.write('\n')
    #     f.write('\n')
    #     f.close()

    #     np.save(os.path.join(folder_path, "test_energy.npy"), test_energy)
    #     np.save(os.path.join(folder_path, "gt.npy"), gt)
    #     np.save(os.path.join(folder_path, "pred.npy"), pred)

    #     return
