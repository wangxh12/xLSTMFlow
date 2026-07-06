import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class MinnesotaPredLoader(Dataset):
    """
    Prediction-based anomaly detection loader for Minnesota / Thor CSV files.

    Return format follows TSLib forecasting style:

        seq_x, seq_y, seq_x_mark, seq_y_mark

    Shapes:
        seq_x:      [seq_len, C]
        seq_y:      [label_len + pred_len, C]
        seq_x_mark: [seq_len, 1]
        seq_y_mark: [label_len + pred_len, 1]

    For anomaly detection:
        seq_y_mark stores labels aligned with seq_y.
        During evaluation, use:
            future_label = seq_y_mark[-pred_len:]
    """

    def __init__(self, args, root_path, win_size=None, flag="train"):
        self.args = args
        self.root_path = root_path
        self.flag = flag

        self.seq_len = getattr(args, "seq_len")
        self.label_len = getattr(args, "label_len", 0)
        self.pred_len = getattr(args, "pred_len")

        if self.pred_len <= 0:
            raise ValueError(
                f"MinnesotaPredLoader requires pred_len > 0, got pred_len={self.pred_len}"
            )

        if self.label_len < 0:
            raise ValueError(
                f"label_len should be >= 0, got label_len={self.label_len}"
            )

        self.target_fields = [
            "navalt", "alt", "h", "navvn", "navve", "navvd",
            "vn", "ve", "vd", "p", "q", "r"
        ]

        self.train_file = getattr(args, "train_file", "ThorFlight104.csv")
        self.test_file = getattr(args, "test_file", "ThorFlight121.csv")

        self.train_path = os.path.join(root_path, self.train_file)
        self.test_path = os.path.join(root_path, self.test_file)

        # train / val 可以用较大 step，test 建议 step=1，方便和 label 对齐
        if flag == "test":
            self.step = 1
        else:
            self.step = getattr(args, "stride", 1)

        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        if "label" not in test_df.columns:
            raise ValueError(f"'label' column not found in {self.test_path}")

        common_fields = [
            f for f in self.target_fields
            if f in train_df.columns and f in test_df.columns
        ]

        if len(common_fields) != len(self.target_fields):
            missing = [f for f in self.target_fields if f not in common_fields]
            raise ValueError(f"Missing target fields: {missing}")

        train_raw = train_df[common_fields].values.astype("float32")
        test_raw = test_df[common_fields].values.astype("float32")
        test_label = test_df["label"].values.astype("float32")

        train_raw = np.nan_to_num(train_raw)
        test_raw = np.nan_to_num(test_raw)

        # 从 ThorFlight104 中切 train / val
        border = int(len(train_raw) * 0.8)
        train_part = train_raw[:border]
        val_part = train_raw[border:]

        scaler_type = getattr(args, "scaler", "standard")

        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaler: {scaler_type}")

        # 只用训练段 fit scaler，避免验证集泄漏
        self.scaler.fit(train_part)

        self.train = self.scaler.transform(train_part).astype("float32")
        self.val = self.scaler.transform(val_part).astype("float32")
        self.test = self.scaler.transform(test_raw).astype("float32")

        # train / val 默认无异常标签
        self.train_labels = np.zeros(len(self.train), dtype="float32")
        self.val_labels = np.zeros(len(self.val), dtype="float32")
        self.test_labels = test_label.astype("float32")

        if flag == "train":
            self.data_x = self.train
            self.data_y = self.train
            self.labels = self.train_labels
        elif flag == "val":
            self.data_x = self.val
            self.data_y = self.val
            self.labels = self.val_labels
        elif flag == "test":
            self.data_x = self.test
            self.data_y = self.test
            self.labels = self.test_labels
        else:
            raise ValueError(f"Unsupported flag: {flag}")

        print(f"[MinnesotaPred] flag={flag}")
        print(f"Train: {self.train.shape}, Val: {self.val.shape}, Test: {self.test.shape}")
        print(f"Fields: {common_fields}")
        print(f"seq_len={self.seq_len}, label_len={self.label_len}, pred_len={self.pred_len}")
        print(f"Step: {self.step}")
        print(f"Scaler: {scaler_type}")

    def __len__(self):
        """
        Need:
            seq_x: data[s_begin : s_begin + seq_len]
            seq_y: data[s_end - label_len : s_end + pred_len]

        The furthest accessed point is:
            s_begin + seq_len + pred_len - 1

        Therefore:
            max_start = len(data) - seq_len - pred_len
        """
        data_len = len(self.data_x)
        max_start = data_len - self.seq_len - self.pred_len

        if max_start < 0:
            return 0

        return max_start // self.step + 1

    def __getitem__(self, index):
        s_begin = index * self.step
        s_end = s_begin + self.seq_len

        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        # 这里暂时没有真实时间特征，seq_x_mark 先给全 0
        # shape: [seq_len, 1]
        seq_x_mark = np.zeros((self.seq_len, 1), dtype="float32")

        # seq_y_mark 用来保存 seq_y 对应的 anomaly label
        # shape: [label_len + pred_len, 1]
        seq_y_label = self.labels[r_begin:r_end]
        seq_y_mark = seq_y_label.reshape(-1, 1).astype("float32")

        return (
            seq_x.astype("float32"),
            seq_y.astype("float32"),
            seq_x_mark.astype("float32"),
            seq_y_mark.astype("float32"),
        )


class MinnesotaSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, flag="train"):
        self.args = args
        self.root_path = root_path
        self.win_size = win_size
        self.flag = flag

        self.target_fields = [
            'navalt', 'alt', 'h', 'navvn', 'navve', 'navvd',
            'vn', 've', 'vd', 'p', 'q', 'r'
        ]

        self.train_file = getattr(args, "train_file", "ThorFlight104.csv")
        self.test_file = getattr(args, "test_file", "ThorFlight121.csv")

        self.train_path = os.path.join(root_path, self.train_file)
        self.test_path = os.path.join(root_path, self.test_file)

        # train / val 可以用较大 stride，test 建议 step=1，方便和 label 对齐
        if flag == "test":
            self.step = 1
        else:
            self.step = getattr(args, "stride", 1)

        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        if "label" not in test_df.columns:
            raise ValueError(f"'label' column not found in {self.test_path}")

        # 保持字段顺序，不要用 set
        common_fields = [
            f for f in self.target_fields
            if f in train_df.columns and f in test_df.columns
        ]

        if len(common_fields) != len(self.target_fields):
            missing = [f for f in self.target_fields if f not in common_fields]
            raise ValueError(f"Missing target fields: {missing}")

        train_raw = train_df[common_fields].values.astype("float32")
        test_raw = test_df[common_fields].values.astype("float32")
        test_label = test_df["label"].values.astype("float32")

        train_raw = np.nan_to_num(train_raw)
        test_raw = np.nan_to_num(test_raw)

        # 从 ThorFlight104 里切 train / val
        border = int(len(train_raw) * 0.8)
        train_part = train_raw[:border]
        val_part = train_raw[border:]

        # 只用训练段 fit scaler，避免验证集泄漏
        # self.scaler = StandardScaler()
        self.scaler = MinMaxScaler()
        self.scaler.fit(train_part)

        self.train = self.scaler.transform(train_part).astype("float32")
        self.val = self.scaler.transform(val_part).astype("float32")
        self.test = self.scaler.transform(test_raw).astype("float32")
        self.test_labels = test_label

        print(f"[Minnesota] flag={flag}")
        print(f"Train: {self.train.shape}, Val: {self.val.shape}, Test: {self.test.shape}")
        print(f"Fields: {common_fields}")
        print(f"Step: {self.step}")

    def __len__(self):
        if self.flag == "train":
            data_len = len(self.train)
        elif self.flag == "val":
            data_len = len(self.val)
        elif self.flag == "test":
            data_len = len(self.test)
        else:
            data_len = len(self.test)

        return (data_len - self.win_size) // self.step + 1

    def __getitem__(self, index):
        s_begin = index * self.step
        s_end = s_begin + self.win_size

        if self.flag == "train":
            x = self.train[s_begin:s_end]
            y = np.zeros(self.win_size, dtype="float32")
        elif self.flag == "val":
            x = self.val[s_begin:s_end]
            y = np.zeros(self.win_size, dtype="float32")
        else:
            x = self.test[s_begin:s_end]
            y = self.test_labels[s_begin:s_end]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)