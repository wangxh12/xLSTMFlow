import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class ALFAReconSegLoader(Dataset):
    """
    ALFA reconstruction dataloader for TSLib anomaly_detection.

    Directory structure:
        root_path/
        ├── train/
        │   ├── xxx.csv
        │   ├── yyy.csv
        │   └── ...
        └── test/
            ├── test_1.csv
            ├── test_2.csv
            └── ...

    CSV columns:
        timestamp_ns,time_s,
        field.angular_velocity.x,field.angular_velocity.y,field.angular_velocity.z,
        field.linear_acceleration.x,field.linear_acceleration.y,field.linear_acceleration.z,
        field.magnetic_field.x,field.magnetic_field.y,field.magnetic_field.z,
        field.fluid_pressure,field.temperature,
        field.measured.pitch,field.measured.roll,field.measured.yaw,
        field.alt_error,field.aspd_error,field.xtrack_error,field.wp_dist,
        label

    Model input:
        all columns except timestamp_ns, time_s, label

    Return:
        seq_x:     [win_size, 18]
        seq_label: [win_size, 1]
    """

    TIME_COLS = ["timestamp_ns", "time_s"]
    LABEL_COL = "label"

    INPUT_COLS = [
        "field.angular_velocity.x",
        "field.angular_velocity.y",
        "field.angular_velocity.z",
        "field.linear_acceleration.x",
        "field.linear_acceleration.y",
        "field.linear_acceleration.z",
        "field.magnetic_field.x",
        "field.magnetic_field.y",
        "field.magnetic_field.z",
        "field.fluid_pressure",
        "field.temperature",
        "field.measured.pitch",
        "field.measured.roll",
        "field.measured.yaw",
        "field.alt_error",
        "field.aspd_error",
        "field.xtrack_error",
        "field.wp_dist",
    ]

    def __init__(
        self,
        args,
        root_path,
        win_size,
        step=1,
        flag="train",
        data_path=None,
        val_ratio=0.1,
        scale=True,
    ):
        assert flag in ["train", "val", "test"], f"Unsupported flag: {flag}"

        self.args = args
        self.root_path = Path(root_path)
        self.win_size = int(win_size)
        self.step = int(step)
        self.flag = flag
        self.data_path = data_path
        self.val_ratio = float(val_ratio)
        self.scale = scale

        self.train_dir = self.root_path / "train"
        self.test_dir = self.root_path / "test"

        if not self.train_dir.exists():
            raise FileNotFoundError(f"Train directory not found: {self.train_dir}")

        if self.flag == "test" and not self.test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {self.test_dir}")

        self.series = []
        self.labels = []
        self.windows = []

        self.scaler = StandardScaler()

        self.__read_data__()

    def __read_data__(self):
        train_files = self._list_csv_files(self.train_dir)
        if len(train_files) == 0:
            raise RuntimeError(f"No csv files found in train directory: {self.train_dir}")

        train_raw = []
        for file_path in train_files:
            x, _ = self._read_one_csv(file_path)
            if len(x) >= self.win_size:
                train_raw.append(x)

        if len(train_raw) == 0:
            raise RuntimeError(
                f"No valid train file has length >= win_size={self.win_size}"
            )

        # Global train statistics: fit scaler using all rows from all train csv files.
        train_concat = np.concatenate(train_raw, axis=0)

        self.scaler.fit(train_concat)

        if self.flag in ["train", "val"]:
            self._build_train_or_val(train_raw)
        else:
            self._build_test()

        if len(self.windows) == 0:
            raise RuntimeError(
                f"No windows generated for flag={self.flag}. "
                f"Check win_size={self.win_size}, step={self.step}, and file lengths."
            )

    def _build_train_or_val(self, train_raw):
        """
        Build train / val windows from train folder.

        Important:
        - scaler is global over all train files.
        - windows do not cross file boundaries.
        - train labels are forced to 0 because this is unsupervised reconstruction training.
        """
        for x in train_raw:
            x = self.scaler.transform(x)

            x = x.astype(np.float32)
            y = np.zeros((len(x),), dtype=np.int64)

            all_starts = np.arange(
                0,
                len(x) - self.win_size + 1,
                self.step,
                dtype=np.int64,
            )

            if len(all_starts) == 0:
                continue

            split = int(len(all_starts) * (1.0 - self.val_ratio))

            if self.flag == "train":
                starts = all_starts[:split]
            else:
                starts = all_starts[split:]

            if len(starts) == 0:
                continue

            sid = len(self.series)
            self.series.append(x)
            self.labels.append(y)

            for s in starts:
                self.windows.append((sid, int(s)))

    def _build_test(self):
        test_file = self._resolve_test_file(self.data_path)
        x, y = self._read_one_csv(test_file)

        if len(x) < self.win_size:
            raise RuntimeError(
                f"Test file is shorter than win_size. "
                f"file={test_file}, length={len(x)}, win_size={self.win_size}"
            )

        x = self.scaler.transform(x)

        x = x.astype(np.float32)
        y = y.astype(np.int64)

        sid = len(self.series)
        self.series.append(x)
        self.labels.append(y)

        starts = np.arange(
            0,
            len(x) - self.win_size + 1,
            self.step,
            dtype=np.int64,
        )

        for s in starts:
            self.windows.append((sid, int(s)))

    def _read_one_csv(self, file_path):
        file_path = Path(file_path)
        df = pd.read_csv(file_path)

        missing_cols = [c for c in self.INPUT_COLS if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing input columns in {file_path}:\n{missing_cols}"
            )

        x_df = df[self.INPUT_COLS].copy()

        for col in self.INPUT_COLS:
            x_df[col] = pd.to_numeric(x_df[col], errors="coerce")

        x_df = x_df.replace([np.inf, -np.inf], np.nan)
        x_df = x_df.interpolate(method="linear", limit_direction="both")
        x_df = x_df.ffill().bfill().fillna(0.0)

        x = x_df.values.astype(np.float32)

        if self.LABEL_COL in df.columns:
            y = pd.to_numeric(df[self.LABEL_COL], errors="coerce")
            y = y.fillna(0).values
            y = (y > 0).astype(np.int64)
        else:
            y = np.zeros((len(df),), dtype=np.int64)

        if len(x) != len(y):
            raise RuntimeError(
                f"Feature length and label length mismatch in {file_path}: "
                f"x={len(x)}, y={len(y)}"
            )

        return x, y

    def _resolve_test_file(self, data_path):
        """
        Resolve one test csv.

        Priority:
        1. data_path argument
        2. environment variable ALFA_TEST_FILE
        3. if root_path/test has exactly one csv, use it
        """
        data_path = data_path or os.environ.get("ALFA_TEST_FILE", None)

        if data_path is not None and str(data_path).strip() != "":
            p = Path(data_path)

            candidates = []
            if p.is_absolute():
                candidates.append(p)
            else:
                candidates.append(self.test_dir / p)
                candidates.append(self.root_path / p)
                candidates.append(p)

            for c in candidates:
                if c.exists() and c.is_file():
                    return c

            raise FileNotFoundError(
                f"Cannot resolve test file from data_path={data_path}. "
                f"Tried: {candidates}"
            )

        test_files = self._list_csv_files(self.test_dir)

        if len(test_files) == 1:
            return test_files[0]

        if len(test_files) == 0:
            raise RuntimeError(f"No csv files found in test directory: {self.test_dir}")

        examples = [p.name for p in test_files[:5]]
        raise RuntimeError(
            "Multiple test csv files found, but no data_path or ALFA_TEST_FILE was given. "
            f"Please specify one test csv. Examples: {examples}"
        )

    @staticmethod
    def _list_csv_files(folder):
        folder = Path(folder)
        return sorted([p for p in folder.glob("*.csv") if p.is_file()])

    def __getitem__(self, index):
        sid, start = self.windows[index]
        end = start + self.win_size

        seq_x = self.series[sid][start:end]
        seq_label = self.labels[sid][start:end].reshape(-1, 1)

        return np.float32(seq_x), np.float32(seq_label)

    def __len__(self):
        return len(self.windows)

    def inverse_transform(self, data):
        if self.scaler is None:
            return data
        return self.scaler.inverse_transform(data)
    
    

class ALFAPredSegLoader(Dataset):
    """
    ALFA prediction dataloader for anomaly detection.

    用 N-1 个变量预测其中 1 个目标变量。

    root_path/
    ├── train/
    │   ├── xxx.csv
    │   └── ...
    └── test/
        ├── xxx.csv
        └── ...

    Train / Val:
        return:
            seq_x:      [seq_len, N-1]
            seq_y:      [pred_len, 1]      # target value
            seq_x_mark: [seq_len, 1]       # dummy
            seq_y_mark: [pred_len, 1]      # dummy

    Test:
        return:
            seq_x:      [seq_len, N-1]
            seq_y:      [pred_len, 1]      # anomaly label
            seq_x_mark: [seq_len, 1]       # dummy
            seq_y_mark: [pred_len, 1]      # true target value, used for residual score
    """

    TIME_COLS = ["timestamp_ns", "time_s"]
    LABEL_COL = "label"

    ALL_FEATURE_COLS = [
        "field.angular_velocity.x",
        "field.angular_velocity.y",
        "field.angular_velocity.z",
        "field.linear_acceleration.x",
        "field.linear_acceleration.y",
        "field.linear_acceleration.z",
        "field.magnetic_field.x",
        "field.magnetic_field.y",
        "field.magnetic_field.z",
        "field.fluid_pressure",
        "field.temperature",
        "field.measured.pitch",
        "field.measured.roll",
        "field.measured.yaw",
        "field.alt_error",
        "field.aspd_error",
        "field.xtrack_error",
        "field.wp_dist",
    ]

    TARGET_ALIAS = {
        "alt_error": "field.alt_error",
        "aspd_error": "field.aspd_error",
        "xtrack_error": "field.xtrack_error",
        "wp_dist": "field.wp_dist",
        "roll": "field.measured.roll",
        "pitch": "field.measured.pitch",
        "yaw": "field.measured.yaw",
        "pressure": "field.fluid_pressure",
        "temperature": "field.temperature",
    }

    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        seq_len=None,
        pred_len=None,
        step=1,
        data_path=None,
        target="field.alt_error",
        val_ratio=0.1,
        scale=True,
        **kwargs,
    ):
        assert flag in ["train", "val", "test"], f"Unsupported flag: {flag}"

        self.root_path = Path(root_path)
        self.flag = flag
        self.step = int(step)
        self.data_path = data_path
        self.val_ratio = float(val_ratio)
        self.scale = scale

        # 兼容 TSLib forecast 风格 size=[seq_len, label_len, pred_len]
        if size is not None:
            self.seq_len = int(size[0])
            self.pred_len = int(size[-1])
        else:
            if seq_len is None:
                seq_len = kwargs.get("win_size", None)
            if pred_len is None:
                pred_len = kwargs.get("pred_len", None)

            if seq_len is None or pred_len is None:
                raise ValueError("seq_len and pred_len must be specified.")

            self.seq_len = int(seq_len)
            self.pred_len = int(pred_len)

        self.target_col = self._resolve_target(target)

        if self.target_col not in self.ALL_FEATURE_COLS:
            raise ValueError(
                f"target={target} is not a valid feature column. "
                f"Available targets: {self.ALL_FEATURE_COLS}"
            )

        self.input_cols = [c for c in self.ALL_FEATURE_COLS if c != self.target_col]

        self.train_dir = self.root_path / "train"
        self.test_dir = self.root_path / "test"

        if not self.train_dir.exists():
            raise FileNotFoundError(f"Train directory not found: {self.train_dir}")

        if self.flag == "test" and not self.test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {self.test_dir}")

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        self.series_x = []
        self.series_y = []
        self.series_label = []
        self.windows = []

        self.__read_data__()

    def __read_data__(self):
        train_files = self._list_csv_files(self.train_dir)
        if len(train_files) == 0:
            raise RuntimeError(f"No csv files found in {self.train_dir}")

        train_x_raw = []
        train_y_raw = []

        for fp in train_files:
            x, y, _ = self._read_one_csv(fp)

            if len(x) >= self.seq_len + self.pred_len:
                train_x_raw.append(x)
                train_y_raw.append(y)

        if len(train_x_raw) == 0:
            raise RuntimeError(
                f"No valid train file with length >= seq_len + pred_len "
                f"({self.seq_len} + {self.pred_len})"
            )

        # 全局统计量：所有 train csv 拼起来拟合 scaler
        train_x_concat = np.concatenate(train_x_raw, axis=0)
        train_y_concat = np.concatenate(train_y_raw, axis=0).reshape(-1, 1)

        if self.scale:
            self.x_scaler.fit(train_x_concat)
            self.y_scaler.fit(train_y_concat)
        else:
            self.x_scaler = None
            self.y_scaler = None

        if self.flag in ["train", "val"]:
            self._build_train_or_val(train_x_raw, train_y_raw)
        else:
            self._build_test()

        if len(self.windows) == 0:
            raise RuntimeError(
                f"No windows generated. flag={self.flag}, "
                f"seq_len={self.seq_len}, pred_len={self.pred_len}, step={self.step}"
            )

    def _build_train_or_val(self, train_x_raw, train_y_raw):
        for x, y in zip(train_x_raw, train_y_raw):
            if self.scale:
                x = self.x_scaler.transform(x)
                y = self.y_scaler.transform(y.reshape(-1, 1)).reshape(-1)

            x = x.astype(np.float32)
            y = y.astype(np.float32)
            label = np.zeros(len(y), dtype=np.int64)

            max_start = len(x) - self.seq_len - self.pred_len + 1
            starts = np.arange(0, max_start, self.step, dtype=np.int64)

            if len(starts) == 0:
                continue

            split = int(len(starts) * (1.0 - self.val_ratio))

            if self.flag == "train":
                selected_starts = starts[:split]
            else:
                selected_starts = starts[split:]

            if len(selected_starts) == 0:
                continue

            sid = len(self.series_x)
            self.series_x.append(x)
            self.series_y.append(y)
            self.series_label.append(label)

            for s in selected_starts:
                self.windows.append((sid, int(s)))

    def _build_test(self):
        test_file = self._resolve_test_file(self.data_path)
        x, y, label = self._read_one_csv(test_file)

        if len(x) < self.seq_len + self.pred_len:
            raise RuntimeError(
                f"Test file is too short. file={test_file}, "
                f"length={len(x)}, required={self.seq_len + self.pred_len}"
            )

        if self.scale:
            x = self.x_scaler.transform(x)
            y = self.y_scaler.transform(y.reshape(-1, 1)).reshape(-1)

        x = x.astype(np.float32)
        y = y.astype(np.float32)
        label = label.astype(np.int64)

        sid = len(self.series_x)
        self.series_x.append(x)
        self.series_y.append(y)
        self.series_label.append(label)

        max_start = len(x) - self.seq_len - self.pred_len + 1
        starts = np.arange(0, max_start, self.step, dtype=np.int64)

        for s in starts:
            self.windows.append((sid, int(s)))

    def _read_one_csv(self, file_path):
        file_path = Path(file_path)
        df = pd.read_csv(file_path)

        required_cols = self.input_cols + [self.target_col]

        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing columns in {file_path}:\n{missing_cols}"
            )

        x_df = df[self.input_cols].copy()
        y_df = df[[self.target_col]].copy()

        for col in self.input_cols:
            x_df[col] = pd.to_numeric(x_df[col], errors="coerce")

        y_df[self.target_col] = pd.to_numeric(y_df[self.target_col], errors="coerce")

        x_df = self._clean_numeric_df(x_df)
        y_df = self._clean_numeric_df(y_df)

        x = x_df.values.astype(np.float32)
        y = y_df[self.target_col].values.astype(np.float32)

        if self.LABEL_COL in df.columns:
            label = pd.to_numeric(df[self.LABEL_COL], errors="coerce")
            label = label.fillna(0).values
            label = (label > 0).astype(np.int64)
        else:
            label = np.zeros(len(df), dtype=np.int64)

        if len(x) != len(y) or len(x) != len(label):
            raise RuntimeError(
                f"Length mismatch in {file_path}: "
                f"x={len(x)}, y={len(y)}, label={len(label)}"
            )

        return x, y, label

    @staticmethod
    def _clean_numeric_df(df):
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.interpolate(method="linear", limit_direction="both")
        df = df.ffill().bfill().fillna(0.0)
        return df

    def _resolve_test_file(self, data_path):
        """
        每次 run.py 只测试一个 csv。

        优先级：
        1. data_path
        2. 环境变量 ALFA_TEST_FILE
        3. test 文件夹下只有一个 csv 时自动使用
        """
        data_path = data_path or os.environ.get("ALFA_TEST_FILE", None)

        if data_path is not None and str(data_path).strip() != "":
            p = Path(data_path)

            candidates = []
            if p.is_absolute():
                candidates.append(p)
            else:
                candidates.append(self.test_dir / p)
                candidates.append(self.root_path / p)
                candidates.append(p)

            for c in candidates:
                if c.exists() and c.is_file():
                    return c

            raise FileNotFoundError(
                f"Cannot resolve test file from data_path={data_path}. "
                f"Tried: {candidates}"
            )

        test_files = self._list_csv_files(self.test_dir)

        if len(test_files) == 1:
            return test_files[0]

        if len(test_files) == 0:
            raise RuntimeError(f"No csv files found in {self.test_dir}")

        examples = [p.name for p in test_files[:5]]
        raise RuntimeError(
            "Multiple test csv files found, but no data_path or ALFA_TEST_FILE was given. "
            f"Please specify one test csv. Examples: {examples}"
        )

    def _resolve_target(self, target):
        target = str(target)
        return self.TARGET_ALIAS.get(target, target)

    @staticmethod
    def _list_csv_files(folder):
        folder = Path(folder)
        return sorted([p for p in folder.glob("*.csv") if p.is_file()])

    def __getitem__(self, index):
        sid, start = self.windows[index]

        x_begin = start
        x_end = start + self.seq_len

        y_begin = x_end
        y_end = x_end + self.pred_len

        seq_x = self.series_x[sid][x_begin:x_end]              # [seq_len, N-1]
        true_y = self.series_y[sid][y_begin:y_end]             # [pred_len]
        label_y = self.series_label[sid][y_begin:y_end]        # [pred_len]

        seq_x = seq_x.astype(np.float32)

        # dummy mark，保持预测式接口一致
        seq_x_mark = np.zeros((self.seq_len, 1), dtype=np.float32)

        if self.flag == "test":
            # 按你的要求：测试阶段 y 返回 label
            seq_y = label_y.reshape(-1, 1).astype(np.float32)

            # 真实目标值放到 seq_y_mark，后续算残差时用
            seq_y_mark = true_y.reshape(-1, 1).astype(np.float32)
        else:
            # 训练/验证阶段 y 返回被预测字段的真实值
            seq_y = true_y.reshape(-1, 1).astype(np.float32)
            seq_y_mark = np.zeros((self.pred_len, 1), dtype=np.float32)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.windows)

    def inverse_transform_y(self, data):
        """
        反归一化目标字段预测值。
        data shape: [*, 1] or [*]
        """
        if self.y_scaler is None:
            return data

        arr = np.asarray(data)
        original_shape = arr.shape
        arr = arr.reshape(-1, 1)
        arr = self.y_scaler.inverse_transform(arr)
        return arr.reshape(original_shape)

    def inverse_transform_x(self, data):
        """
        反归一化输入变量。
        data shape: [*, N-1]
        """
        if self.x_scaler is None:
            return data
        return self.x_scaler.inverse_transform(data)