import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class AlfaPredLoader(Dataset):
    """
    ALFA predictive anomaly detection dataset.

    训练:
        data/alfa/train/*.csv
        全部 no_failure，label 全 0

    测试:
        data/alfa/test/*.csv
        每次只测试一个文件，通过 args.alfa_test_file 指定

    支持两种模式:
        1. all-to-all:
            18 维输入 -> 18 维预测
            --alfa_target all
            --enc_in 18 --c_out 18

        2. multivariate-to-single:
            18 维输入 -> 1 维预测
            --alfa_target field.alt_error
            --enc_in 18 --c_out 1
    """

    feature_cols = [
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
        root_path,
        flag="train",
        size=None,
        win_size=None,
        pred_len=None,
        step=1,
        args=None,
    ):
        super().__init__()

        assert flag in ["train", "val", "test"]
        self.root_path = root_path
        self.flag = flag
        self.args = args
        self.step = step

        if size is not None:
            self.seq_len = int(size[0])
            self.pred_len = int(size[1])
        else:
            self.seq_len = int(win_size)
            self.pred_len = int(pred_len)

        self.win_size = self.seq_len

        target = getattr(args, "alfa_target", "all") if args is not None else "all"

        if target == "all":
            self.target_cols = self.feature_cols
        else:
            self.target_cols = [x.strip() for x in target.split(",") if x.strip()]

        self.enc_in = len(self.feature_cols)
        self.c_out = len(self.target_cols)

        train_dir = os.path.join(root_path, "train")
        test_dir = os.path.join(root_path, "test")

        all_train_files = sorted(glob.glob(os.path.join(train_dir, "*.csv")))
        all_test_files = sorted(glob.glob(os.path.join(test_dir, "*.csv")))

        if len(all_train_files) == 0:
            raise RuntimeError(f"No train csv files found in {train_dir}")

        if len(all_test_files) == 0:
            raise RuntimeError(f"No test csv files found in {test_dir}")

        # ==========================================================
        # 文件级划分 train / val，避免同一个 flight 的窗口同时进 train 和 val
        # ==========================================================
        train_ratio = float(getattr(args, "alfa_train_ratio", 0.8)) if args is not None else 0.8
        split = int(len(all_train_files) * train_ratio)
        split = min(max(split, 1), len(all_train_files) - 1)

        fit_files = all_train_files[:split]

        if flag == "train":
            self.files = all_train_files[:split]
        elif flag == "val":
            self.files = all_train_files[split:]
        else:
            test_file = getattr(args, "alfa_test_file", None) if args is not None else None

            if test_file is None or test_file == "":
                # 默认取第一个，真正批量测试时会在 exp.test() 里逐个设置 alfa_test_file
                self.files = [all_test_files[0]]
            else:
                file_path = os.path.join(test_dir, test_file)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"ALFA test file not found: {file_path}")
                self.files = [file_path]

        # ==========================================================
        # scaler 只在训练 split 上 fit，不能使用 val/test
        # ==========================================================
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        fit_x_list = []
        fit_y_list = []

        for fp in fit_files:
            df = self._read_csv(fp)
            fit_x_list.append(df[self.feature_cols].values)
            fit_y_list.append(df[self.target_cols].values)

        fit_x = np.concatenate(fit_x_list, axis=0)
        fit_y = np.concatenate(fit_y_list, axis=0)

        self.scaler_x.fit(fit_x)
        self.scaler_y.fit(fit_y)

        # ==========================================================
        # 读取当前 flag 对应文件，不允许跨文件构造窗口
        # ==========================================================
        self.x_list = []
        self.y_list = []
        self.label_list = []
        self.raw_lens = []
        self.file_names = []

        self.indices = []

        for file_id, fp in enumerate(self.files):
            df = self._read_csv(fp)

            x = df[self.feature_cols].values.astype(np.float32)
            y = df[self.target_cols].values.astype(np.float32)

            if "label" in df.columns:
                label = df["label"].values.astype(np.float32)
                label = (label > 0).astype(np.float32)
            else:
                label = np.zeros(len(df), dtype=np.float32)

            x = self.scaler_x.transform(x).astype(np.float32)
            y = self.scaler_y.transform(y).astype(np.float32)

            self.x_list.append(x)
            self.y_list.append(y)
            self.label_list.append(label)
            self.raw_lens.append(len(df))
            self.file_names.append(os.path.basename(fp))

            max_start = len(df) - self.seq_len - self.pred_len + 1

            if max_start <= 0:
                continue

            for start in range(0, max_start, self.step):
                # train/val 理论上都是正常文件，不过这里再保险一次：
                # 训练和验证只取完整正常窗口
                if flag in ["train", "val"]:
                    full_label = label[start:start + self.seq_len + self.pred_len]
                    if full_label.max() > 0:
                        continue

                self.indices.append((file_id, start))

        if len(self.indices) == 0:
            raise RuntimeError(
                f"No valid windows for ALFA flag={flag}, files={self.files}"
            )

        # 兼容你当前 exp 里的 raw_test_len = len(test_data.test)
        # test 时只加载一个文件，所以这里成立
        if flag == "test":
            self.test = self.x_list[0]
        else:
            self.test = None

    def _read_csv(self, fp):
        df = pd.read_csv(fp)

        required_cols = self.feature_cols + ["label"]
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            raise RuntimeError(f"{fp} missing columns: {missing}")

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill().fillna(0)

        return df

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        file_id, start = self.indices[index]

        s_begin = start
        s_end = s_begin + self.seq_len

        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.x_list[file_id][s_begin:s_end]
        seq_y = self.y_list[file_id][r_begin:r_end]
        seq_label = self.label_list[file_id][r_begin:r_end]

        return seq_x, seq_y, seq_label


class ThorNavAltPredLoader(Dataset):
    """
    Thor 预测式异常检测 Dataset.

    训练集:
        data/ThorFlight104.csv

    测试集:
        data/ThorFlight121.csv

    输入:
        past seq_len steps:
        ["alt", "h", "navvn", "navve", "navvd",
         "vn", "ve", "vd", "p", "q", "r"]

    预测:
        future pred_len steps:
        ["navalt"]

    label:
        both train and test csv files have label column.
        train/val only use windows whose input+future labels are all 0.
        test uses all windows.
    """

    def __init__(self, args, root_path, win_size, pred_len, step=1, flag="train"):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.pred_len = pred_len

        self.train_file = args.train_file
        self.test_file = args.test_file

        self.input_fields = [
            "alt", "h", "navvn", "navve", "navvd",
            "vn", "ve", "vd", "p", "q", "r"
        ]

        self.target_fields = ["navalt"]
        self.label_field = "label"

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        train_path = os.path.join(self.root_path, self.train_file)
        test_path = os.path.join(self.root_path, self.test_file)

        assert os.path.exists(train_path), f"missing train file: {train_path}"
        assert os.path.exists(test_path), f"missing test file: {test_path}"

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        self._check_columns(train_df, train_path)
        self._check_columns(test_df, test_path)

        train_x, train_y, train_label = self._read_xy_label(train_df)
        test_x, test_y, test_label = self._read_xy_label(test_df)

        # 只用训练集正常点拟合归一化参数
        normal_mask = train_label.reshape(-1) == 0
        assert normal_mask.sum() > 0, "ThorFlight104.csv has no normal points with label == 0"

        self.scaler_x.fit(train_x[normal_mask])
        self.scaler_y.fit(train_y[normal_mask])

        train_x = self.scaler_x.transform(train_x)
        train_y = self.scaler_y.transform(train_y)

        test_x = self.scaler_x.transform(test_x)
        test_y = self.scaler_y.transform(test_y)

        # 按时间划分 train / val
        train_len = len(train_x)
        val_start = int(train_len * 0.8)

        self.train_x = train_x[:val_start]
        self.train_y = train_y[:val_start]
        self.train_labels = train_label[:val_start].astype(np.float32).reshape(-1, 1)

        self.val_x = train_x[val_start:]
        self.val_y = train_y[val_start:]
        self.val_labels = train_label[val_start:].astype(np.float32).reshape(-1, 1)

        self.test_x = test_x
        self.test_y = test_y
        self.test_labels = test_label.astype(np.float32).reshape(-1, 1)

        # 训练/验证只使用全正常窗口
        self.train_indices = self._make_indices(self.train_labels, normal_only=True)
        self.val_indices = self._make_indices(self.val_labels, normal_only=True)

        # 测试保留所有窗口
        self.test_indices = self._make_indices(self.test_labels, normal_only=False)

        # 兼容 exp_anomaly_detection_pred.py 里的 raw_test_len = len(test_data.test)
        self.train = self.train_x
        self.val = self.val_x
        self.test = self.test_x

        print("ThorNavAltPred train_x:", self.train_x.shape)
        print("ThorNavAltPred train_y:", self.train_y.shape)
        print("ThorNavAltPred train_labels:", self.train_labels.shape)
        print("ThorNavAltPred train windows:", len(self.train_indices))

        print("ThorNavAltPred val_x:", self.val_x.shape)
        print("ThorNavAltPred val_y:", self.val_y.shape)
        print("ThorNavAltPred val_labels:", self.val_labels.shape)
        print("ThorNavAltPred val windows:", len(self.val_indices))

        print("ThorNavAltPred test_x:", self.test_x.shape)
        print("ThorNavAltPred test_y:", self.test_y.shape)
        print("ThorNavAltPred test_labels:", self.test_labels.shape)
        print("ThorNavAltPred test windows:", len(self.test_indices))

    def _check_columns(self, df, path):
        missing_inputs = [c for c in self.input_fields if c not in df.columns]
        missing_targets = [c for c in self.target_fields if c not in df.columns]

        assert len(missing_inputs) == 0, (
            f"missing input columns in {path}: {missing_inputs}"
        )

        assert len(missing_targets) == 0, (
            f"missing target columns in {path}: {missing_targets}"
        )

        assert self.label_field in df.columns, (
            f"missing label column '{self.label_field}' in {path}"
        )

    def _read_xy_label(self, df):
        x = df[self.input_fields].values.astype(np.float32)
        y = df[self.target_fields].values.astype(np.float32)
        label = df[self.label_field].values.astype(np.float32)

        x = np.nan_to_num(x)
        y = np.nan_to_num(y)
        label = np.nan_to_num(label)

        label = (label > 0).astype(np.float32)

        return x, y, label

    def _make_indices(self, labels, normal_only):
        """
        生成可用窗口起点。

        对于训练/验证:
            normal_only=True
            要求 input window + future patch 全部 label==0。

        对于测试:
            normal_only=False
            保留所有窗口。
        """
        labels = labels.reshape(-1)

        max_start = len(labels) - self.win_size - self.pred_len
        if max_start < 0:
            return []

        indices = []

        for start in range(0, max_start + 1, self.step):
            s_begin = start
            s_end = s_begin + self.win_size

            r_begin = s_end
            r_end = r_begin + self.pred_len

            if normal_only:
                window_label = labels[s_begin:r_end]
                if window_label.max() > 0:
                    continue

            indices.append(start)

        return indices

    def _select_data(self):
        if self.flag == "train":
            return self.train_x, self.train_y, self.train_labels, self.train_indices
        elif self.flag == "val":
            return self.val_x, self.val_y, self.val_labels, self.val_indices
        elif self.flag == "test":
            return self.test_x, self.test_y, self.test_labels, self.test_indices
        else:
            raise ValueError(f"unknown flag: {self.flag}")

    def __len__(self):
        _, _, _, indices = self._select_data()
        return len(indices)

    def __getitem__(self, index):
        data_x, data_y, labels, indices = self._select_data()

        start = indices[index]

        s_begin = start
        s_end = s_begin + self.win_size

        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = data_x[s_begin:s_end]
        seq_y = data_y[r_begin:r_end]
        label_y = labels[r_begin:r_end].reshape(-1)

        return np.float32(seq_x), np.float32(seq_y), np.float32(label_y)

class PredNavAltSegLoader(Dataset):
    """
    预测式异常检测 Dataset。

    输入:
        X = ["alt", "h", "navvn", "navve", "navvd",
             "vn", "ve", "vd", "p", "q", "r"]

    预测:
        Y = ["navalt"]

    返回:
        seq_x:   [seq_len, 11]
        seq_y:   [pred_len, 1]
        label_y: [pred_len]
    """

    def __init__(self, args, root_path, win_size, pred_len, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.pred_len = pred_len

        self.input_fields = [
            "alt", "h", "navvn", "navve", "navvd",
            "vn", "ve", "vd", "p", "q", "r"
        ]

        self.target_fields = ["navalt"]

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        train_path = os.path.join(root_path, "train.csv")
        test_path = os.path.join(root_path, "test.csv")
        label_path = os.path.join(root_path, "test_label.csv")

        assert os.path.exists(train_path), f"missing file: {train_path}"
        assert os.path.exists(test_path), f"missing file: {test_path}"
        assert os.path.exists(label_path), f"missing file: {label_path}"

        train_x, train_y = self._read_xy_csv(train_path)
        test_x, test_y = self._read_xy_csv(test_path)
        test_label = self._read_label_csv(label_path)

        self.scaler_x.fit(train_x)
        self.scaler_y.fit(train_y)

        train_x = self.scaler_x.transform(train_x)
        test_x = self.scaler_x.transform(test_x)

        train_y = self.scaler_y.transform(train_y)
        test_y = self.scaler_y.transform(test_y)

        data_len = len(train_x)
        val_start = int(data_len * 0.8)

        self.train_x = train_x[:val_start]
        self.train_y = train_y[:val_start]

        self.val_x = train_x[val_start:]
        self.val_y = train_y[val_start:]

        self.test_x = test_x
        self.test_y = test_y

        self.train_labels = np.zeros((len(self.train_x), 1), dtype=np.float32)
        self.val_labels = np.zeros((len(self.val_x), 1), dtype=np.float32)
        self.test_labels = test_label.astype(np.float32).reshape(-1, 1)

        # 为了兼容 exp_anomaly_detection_pred.py 里的 len(test_data.test)
        self.train = self.train_x
        self.val = self.val_x
        self.test = self.test_x

        assert len(self.test_x) == len(self.test_labels), \
            f"test length {len(self.test_x)} != label length {len(self.test_labels)}"

        print("PredNavAlt train_x:", self.train_x.shape)
        print("PredNavAlt train_y:", self.train_y.shape)
        print("PredNavAlt val_x:", self.val_x.shape)
        print("PredNavAlt val_y:", self.val_y.shape)
        print("PredNavAlt test_x:", self.test_x.shape)
        print("PredNavAlt test_y:", self.test_y.shape)

    def _read_xy_csv(self, path):
        df = pd.read_csv(path)

        missing_inputs = [c for c in self.input_fields if c not in df.columns]
        missing_targets = [c for c in self.target_fields if c not in df.columns]

        assert len(missing_inputs) == 0, f"missing input columns in {path}: {missing_inputs}"
        assert len(missing_targets) == 0, f"missing target columns in {path}: {missing_targets}"

        x = df[self.input_fields].values.astype(np.float32)
        y = df[self.target_fields].values.astype(np.float32)

        x = np.nan_to_num(x)
        y = np.nan_to_num(y)

        return x, y

    def _read_label_csv(self, path):
        df = pd.read_csv(path)

        # 优先找 label 这一列
        if "label" in df.columns:
            values = df["label"].values
        else:
            df = df.select_dtypes(include=[np.number])
            assert df.shape[1] > 0, f"no numeric label columns found in {path}"
            values = df.values
            if values.ndim == 2:
                values = values.max(axis=1)

        values = (values > 0).astype(np.float32)
        return values

    def _select_data(self):
        if self.flag == "train":
            return self.train_x, self.train_y, self.train_labels
        elif self.flag == "val":
            return self.val_x, self.val_y, self.val_labels
        elif self.flag == "test":
            return self.test_x, self.test_y, self.test_labels
        else:
            raise ValueError(f"unknown flag: {self.flag}")

    def __len__(self):
        data_x, data_y, labels = self._select_data()
        return (len(data_x) - self.win_size - self.pred_len) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step

        data_x, data_y, labels = self._select_data()

        s_begin = index
        s_end = s_begin + self.win_size

        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = data_x[s_begin:s_end]
        seq_y = data_y[r_begin:r_end]
        label_y = labels[r_begin:r_end].reshape(-1)

        return np.float32(seq_x), np.float32(seq_y), np.float32(label_y)

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