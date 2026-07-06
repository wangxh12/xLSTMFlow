import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class UAVTargetFlowSegLoader(Dataset):
    """
    xLSTM + TimeFlow 用的预测式异常检测 Dataset。

    目录结构建议:

        root_path/
        ├── train/
        │   ├── normal_1.csv
        │   └── normal_2.csv
        ├── val/
        │   └── normal_val.csv
        └── test/
            └── fault_1.csv

    每个 CSV:
        timestamp, feature_1, feature_2, ..., target_col, label

    输入:
        x: [seq_len, enc_in]
           除目标飞参 target_col 外的其他飞参

    输出:
        y: [pred_len]
           target_col 的未来 pred_len 个时间步

    label:
        y_label: [pred_len]
           未来 pred_len 个点的异常标签
    """

    def __init__(self, args, flag):
        super().__init__()

        assert flag in ["train", "val", "test"]

        self.args = args
        self.flag = flag

        self.root_path = Path(args.root_path)
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.stride = getattr(args, "stride", 1)

        self.target_col = args.target_col
        self.label_col = getattr(args, "label_col", "label")

        self.timestamp_cols = {
            "timestamp",
            "timestamp_ns",
            "time",
            "time_s",
            "date",
        }

        self.file_paths = self._collect_files(flag)
        self.train_file_paths = self._collect_files("train")

        if len(self.file_paths) == 0:
            raise RuntimeError(f"No CSV files found for flag={flag} under {self.root_path}")

        if len(self.train_file_paths) == 0:
            raise RuntimeError(f"No train CSV files found under {self.root_path / 'train'}")

        self.feature_cols = self._get_feature_cols()
        self.enc_in = len(self.feature_cols)

        if hasattr(args, "enc_in") and args.enc_in != self.enc_in:
            raise ValueError(
                f"args.enc_in={args.enc_in}, but inferred enc_in={self.enc_in}. "
                f"feature_cols={self.feature_cols}"
            )

        (
            self.feature_mean,
            self.feature_std,
            self.target_mean,
            self.target_std,
        ) = self._fit_train_scaler()

        self.data_x = []
        self.data_y = []
        self.data_label = []
        self.file_offsets = []

        self._load_all_files()

        self.indices = self._build_indices()

        self.total_len = sum(len(x) for x in self.data_y)
        self.global_labels = self._build_global_labels()

        print(
            f"[{flag}] files={len(self.file_paths)}, "
            f"samples={len(self.indices)}, "
            f"enc_in={self.enc_in}, "
            f"target={self.target_col}"
        )

    def _collect_files(self, flag):
        dir_path = self.root_path / flag

        if not dir_path.exists():
            # 如果没有 val 文件夹，可以先用 train 代替 val
            if flag == "val":
                dir_path = self.root_path / "train"
            else:
                raise RuntimeError(f"Directory not found: {dir_path}")

        files = sorted(list(dir_path.glob("*.csv")))

        return files

    def _get_feature_cols(self):
        """
        默认:
            所有数值列 - timestamp列 - label列 - target_col
        也可以通过 args.feature_cols 手动传入逗号分隔字符串。
        """
        alfa_col = ['field.angular_velocity.x', 
                'field.angular_velocity.y', 
                'field.angular_velocity.z', 
                'field.linear_acceleration.x', 
                'field.linear_acceleration.y', 
                'field.linear_acceleration.z', 
                'field.magnetic_field.x', 
                'field.magnetic_field.y', 
                'field.magnetic_field.z', 
                'field.fluid_pressure', 
                'field.temperature', 
                'field.measured.pitch', 
                'field.measured.roll', 
                'field.measured.yaw', 
                'field.alt_error', 
                'field.aspd_error', 
                'field.xtrack_error', 
                'field.wp_dist'
                ]
        alfa_col.remove(self.target_col)
        return alfa_col

        feature_cols_arg = getattr(self.args, "feature_cols", None)

        if feature_cols_arg is not None and str(feature_cols_arg).strip() != "":
            feature_cols = [x.strip() for x in feature_cols_arg.split(",")]
            return feature_cols

        df = pd.read_csv(self.train_file_paths[0], nrows=10)

        numeric_cols = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)

        drop_cols = set(self.timestamp_cols)
        drop_cols.add(self.label_col)
        drop_cols.add(self.target_col)

        feature_cols = [c for c in numeric_cols if c not in drop_cols]

        if len(feature_cols) == 0:
            raise RuntimeError("No feature columns inferred. Please set --feature_cols manually.")

        return feature_cols

    def _fit_train_scaler(self):
        xs = []
        ys = []

        for path in self.train_file_paths:
            df = pd.read_csv(path)

            missing = [c for c in self.feature_cols + [self.target_col] if c not in df.columns]
            if len(missing) > 0:
                raise RuntimeError(f"{path} missing columns: {missing}")

            x = df[self.feature_cols].values.astype(np.float32)
            y = df[self.target_col].values.astype(np.float32)

            xs.append(x)
            ys.append(y)

        x_all = np.concatenate(xs, axis=0)
        y_all = np.concatenate(ys, axis=0)

        feature_mean = x_all.mean(axis=0, keepdims=True)
        feature_std = x_all.std(axis=0, keepdims=True)
        feature_std = np.maximum(feature_std, 1e-6)

        target_mean = y_all.mean()
        target_std = y_all.std()
        target_std = max(float(target_std), 1e-6)

        return (
            feature_mean.astype(np.float32),
            feature_std.astype(np.float32),
            float(target_mean),
            float(target_std),
        )

    def _load_all_files(self):
        offset = 0

        for path in self.file_paths:
            df = pd.read_csv(path)

            missing = [c for c in self.feature_cols + [self.target_col] if c not in df.columns]
            if len(missing) > 0:
                raise RuntimeError(f"{path} missing columns: {missing}")

            x = df[self.feature_cols].values.astype(np.float32)
            y = df[self.target_col].values.astype(np.float32)

            x = (x - self.feature_mean) / self.feature_std
            y = (y - self.target_mean) / self.target_std

            if self.label_col in df.columns:
                label = df[self.label_col].values.astype(np.int64)
            else:
                label = np.zeros(len(df), dtype=np.int64)

            self.file_offsets.append(offset)
            offset += len(df)

            self.data_x.append(x)
            self.data_y.append(y)
            self.data_label.append(label)

    def _build_indices(self):
        indices = []

        for file_id, y in enumerate(self.data_y):
            n = len(y)
            max_start = n - self.seq_len - self.pred_len + 1

            if max_start <= 0:
                continue

            for s in range(0, max_start, self.stride):
                indices.append((file_id, s))

        return indices

    def _build_global_labels(self):
        labels = []

        for label in self.data_label:
            labels.append(label)

        return np.concatenate(labels, axis=0)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        file_id, s = self.indices[index]

        x_arr = self.data_x[file_id]
        y_arr = self.data_y[file_id]
        label_arr = self.data_label[file_id]

        t0 = s + self.seq_len
        t1 = t0 + self.pred_len

        batch_x = x_arr[s:t0]             # [32, enc_in]
        batch_y = y_arr[t0:t1]            # [4]
        batch_label = label_arr[t0:t1]    # [4]

        offset = self.file_offsets[file_id]
        batch_index = np.arange(t0, t1) + offset  # [4]

        return (
            torch.from_numpy(batch_x).float(),
            torch.from_numpy(batch_y).float(),
            torch.from_numpy(batch_label).long(),
            torch.from_numpy(batch_index).long(),
        )