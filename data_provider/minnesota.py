import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler


import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class ThorNavAltFlowLoader(Dataset):
    """
    Thor navalt 条件预测式异常检测 Dataset.

    任务定义:
        用过去 32 个时间步的其他飞参预测未来 4 个时间步的 navalt。

    输入:
        seq_x: [win_size, enc_in]
            默认 win_size = 32
            enc_in = 11
            输入字段:
                ["alt", "h", "navvn", "navve", "navvd",
                 "vn", "ve", "vd", "p", "q", "r"]

    预测目标:
        seq_y: [pred_len]
            默认 pred_len = 4
            目标字段:
                ["navalt"]

    标签:
        label_y: [pred_len]
            未来 4 个 navalt 点对应的异常标签。

    索引:
        index_y: [pred_len]
            未来 4 个 navalt 点在当前 split 序列中的位置。
            用于把窗口级 / horizon 级分数聚合回点级分数。

    训练/验证:
        只使用 input window + future target 全部 label==0 的窗口。

    测试:
        保留所有窗口。
    """

    def __init__(
        self,
        args,
        root_path,
        win_size,
        step=1,
        flag="train",
    ):
        super().__init__()

        assert flag in ["train", "val", "test"], f"unknown flag: {flag}"

        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.pred_len = args.pred_len

        self.train_file = args.train_file
        self.test_file = args.test_file

        # =========================
        # 1. 输入字段：除目标飞参 navalt 外的其他飞参
        # =========================

        self.input_fields = [
            "alt", "h", "navvn", "navve", "navvd",
            "vn", "ve", "vd", "p", "q", "r",
        ]

        # 目标飞参
        self.target_field = "navalt"

        # 标签字段
        self.label_field = "label"

        self.enc_in = len(self.input_fields)

        # 如果 args 里有 enc_in，就检查一下，避免配置写错
        if hasattr(args, "enc_in"):
            assert args.enc_in == self.enc_in, (
                f"args.enc_in={args.enc_in}, but inferred enc_in={self.enc_in}. "
                f"input_fields={self.input_fields}"
            )

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

        # =========================
        # 2. 只用训练文件里的正常点拟合 scaler
        # =========================

        normal_mask = train_label.reshape(-1) == 0

        assert normal_mask.sum() > 0, (
            f"{self.train_file} has no normal points with label == 0"
        )

        self.scaler_x.fit(train_x[normal_mask])
        self.scaler_y.fit(train_y[normal_mask])

        train_x = self.scaler_x.transform(train_x).astype(np.float32)
        train_y = self.scaler_y.transform(train_y).astype(np.float32)

        test_x = self.scaler_x.transform(test_x).astype(np.float32)
        test_y = self.scaler_y.transform(test_y).astype(np.float32)

        # =========================
        # 3. 按时间划分 train / val
        # =========================

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

        # =========================
        # 4. 训练/验证只使用全正常窗口，测试保留所有窗口
        # =========================

        self.train_indices = self._make_indices(self.train_labels, normal_only=True)
        self.val_indices = self._make_indices(self.val_labels, normal_only=True)
        self.test_indices = self._make_indices(self.test_labels, normal_only=False)

        # 兼容一些 TSLib 风格代码
        self.train = self.train_x
        self.val = self.val_x
        self.test = self.test_x

        # 当前 flag 对应的数据，用于 exp 里做点级聚合
        cur_x, cur_y, cur_labels, cur_indices = self._select_data()

        self.total_len = len(cur_y)
        self.global_labels = cur_labels.reshape(-1).astype(np.int64)
        self.indices = cur_indices

        print("=" * 80)
        print(f"ThorNavAltFlowLoader flag: {self.flag}")
        print(f"input_fields: {self.input_fields}")
        print(f"target_field: {self.target_field}")
        print(f"enc_in: {self.enc_in}")
        print(f"win_size: {self.win_size}")
        print(f"pred_len: {self.pred_len}")
        print(f"step: {self.step}")

        print("train_x:", self.train_x.shape)
        print("train_y:", self.train_y.shape)
        print("train_labels:", self.train_labels.shape)
        print("train windows:", len(self.train_indices))

        print("val_x:", self.val_x.shape)
        print("val_y:", self.val_y.shape)
        print("val_labels:", self.val_labels.shape)
        print("val windows:", len(self.val_indices))

        print("test_x:", self.test_x.shape)
        print("test_y:", self.test_y.shape)
        print("test_labels:", self.test_labels.shape)
        print("test windows:", len(self.test_indices))
        print("=" * 80)

    def _check_columns(self, df, path):
        missing_inputs = [c for c in self.input_fields if c not in df.columns]
        assert len(missing_inputs) == 0, (
            f"missing input columns in {path}: {missing_inputs}"
        )

        assert self.target_field in df.columns, (
            f"missing target column '{self.target_field}' in {path}"
        )

        assert self.label_field in df.columns, (
            f"missing label column '{self.label_field}' in {path}"
        )

    def _read_xy_label(self, df):
        x = df[self.input_fields].values.astype(np.float32)

        # y 保持成 [N, 1]，方便 StandardScaler
        y = df[[self.target_field]].values.astype(np.float32)

        label = df[self.label_field].values.astype(np.float32)

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        label = np.nan_to_num(label, nan=0.0, posinf=0.0, neginf=0.0)

        label = (label > 0).astype(np.float32)

        return x, y, label

    def _make_indices(self, labels, normal_only):
        """
        生成可用窗口起点。

        对于训练/验证:
            normal_only=True
            要求 input window + future target 全部 label==0。

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
            return (
                self.train_x,
                self.train_y,
                self.train_labels,
                self.train_indices,
            )

        elif self.flag == "val":
            return (
                self.val_x,
                self.val_y,
                self.val_labels,
                self.val_indices,
            )

        elif self.flag == "test":
            return (
                self.test_x,
                self.test_y,
                self.test_labels,
                self.test_indices,
            )

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

        # 输入：过去 32 个时间步的其他飞参
        seq_x = data_x[s_begin:s_end]          # [32, enc_in]

        # 目标：未来 4 个时间步的 navalt
        # data_y 原本是 [N, 1]，这里压成 [4]
        seq_y = data_y[r_begin:r_end, 0]       # [4]

        # 未来 4 个点的标签
        label_y = labels[r_begin:r_end, 0]     # [4]

        # 未来 4 个点在当前 split 序列中的索引
        index_y = np.arange(r_begin, r_end)    # [4]

        return (
            np.float32(seq_x),
            np.float32(seq_y),
            np.float32(label_y),
            np.int64(index_y),
        )


# class ThorNavAltPredLoader(Dataset):
#     """
#     Thor 预测式异常检测 Dataset.

#     训练集:
#         data/ThorFlight104.csv

#     测试集:
#         data/ThorFlight121.csv

#     输入:
#         past seq_len steps:
#         ["alt", "h", "navvn", "navve", "navvd",
#          "vn", "ve", "vd", "p", "q", "r"]

#     预测:
#         future pred_len steps:
#         ["navalt"]

#     label:
#         both train and test csv files have label column.
#         train/val only use windows whose input+future labels are all 0.
#         test uses all windows.
#     """

#     def __init__(self, args, root_path, win_size, step=1, flag="train"):
#         self.args = args
#         self.root_path = root_path
#         self.flag = flag
#         self.step = step
#         self.win_size = win_size
#         self.pred_len = args.pred_len

#         self.train_file = args.train_file
#         self.test_file = args.test_file

#         self.input_fields = [
#             "alt", "h", "navvn", "navve", "navvd",
#             "vn", "ve", "vd", "p", "q", "r"
#         ]

#         self.target_fields = ["navalt"]
#         self.label_field = "label"

#         self.scaler_x = StandardScaler()
#         self.scaler_y = StandardScaler()

#         train_path = os.path.join(self.root_path, self.train_file)
#         test_path = os.path.join(self.root_path, self.test_file)

#         assert os.path.exists(train_path), f"missing train file: {train_path}"
#         assert os.path.exists(test_path), f"missing test file: {test_path}"

#         train_df = pd.read_csv(train_path)
#         test_df = pd.read_csv(test_path)

#         self._check_columns(train_df, train_path)
#         self._check_columns(test_df, test_path)

#         train_x, train_y, train_label = self._read_xy_label(train_df)
#         test_x, test_y, test_label = self._read_xy_label(test_df)

#         # 只用训练集正常点拟合归一化参数
#         normal_mask = train_label.reshape(-1) == 0
#         assert normal_mask.sum() > 0, "ThorFlight104.csv has no normal points with label == 0"

#         self.scaler_x.fit(train_x[normal_mask])
#         self.scaler_y.fit(train_y[normal_mask])

#         train_x = self.scaler_x.transform(train_x)
#         train_y = self.scaler_y.transform(train_y)

#         test_x = self.scaler_x.transform(test_x)
#         test_y = self.scaler_y.transform(test_y)

#         # 按时间划分 train / val
#         train_len = len(train_x)
#         val_start = int(train_len * 0.8)

#         self.train_x = train_x[:val_start]
#         self.train_y = train_y[:val_start]
#         self.train_labels = train_label[:val_start].astype(np.float32).reshape(-1, 1)

#         self.val_x = train_x[val_start:]
#         self.val_y = train_y[val_start:]
#         self.val_labels = train_label[val_start:].astype(np.float32).reshape(-1, 1)

#         self.test_x = test_x
#         self.test_y = test_y
#         self.test_labels = test_label.astype(np.float32).reshape(-1, 1)

#         # 训练/验证只使用全正常窗口
#         self.train_indices = self._make_indices(self.train_labels, normal_only=True)
#         self.val_indices = self._make_indices(self.val_labels, normal_only=True)

#         # 测试保留所有窗口
#         self.test_indices = self._make_indices(self.test_labels, normal_only=False)

#         # 兼容 exp_anomaly_detection_pred.py 里的 raw_test_len = len(test_data.test)
#         self.train = self.train_x
#         self.val = self.val_x
#         self.test = self.test_x

#         print("ThorNavAltPred train_x:", self.train_x.shape)
#         print("ThorNavAltPred train_y:", self.train_y.shape)
#         print("ThorNavAltPred train_labels:", self.train_labels.shape)
#         print("ThorNavAltPred train windows:", len(self.train_indices))

#         print("ThorNavAltPred val_x:", self.val_x.shape)
#         print("ThorNavAltPred val_y:", self.val_y.shape)
#         print("ThorNavAltPred val_labels:", self.val_labels.shape)
#         print("ThorNavAltPred val windows:", len(self.val_indices))

#         print("ThorNavAltPred test_x:", self.test_x.shape)
#         print("ThorNavAltPred test_y:", self.test_y.shape)
#         print("ThorNavAltPred test_labels:", self.test_labels.shape)
#         print("ThorNavAltPred test windows:", len(self.test_indices))

#     def _check_columns(self, df, path):
#         missing_inputs = [c for c in self.input_fields if c not in df.columns]
#         missing_targets = [c for c in self.target_fields if c not in df.columns]

#         assert len(missing_inputs) == 0, (
#             f"missing input columns in {path}: {missing_inputs}"
#         )

#         assert len(missing_targets) == 0, (
#             f"missing target columns in {path}: {missing_targets}"
#         )

#         assert self.label_field in df.columns, (
#             f"missing label column '{self.label_field}' in {path}"
#         )

#     def _read_xy_label(self, df):
#         x = df[self.input_fields].values.astype(np.float32)
#         y = df[self.target_fields].values.astype(np.float32)
#         label = df[self.label_field].values.astype(np.float32)

#         x = np.nan_to_num(x)
#         y = np.nan_to_num(y)
#         label = np.nan_to_num(label)

#         label = (label > 0).astype(np.float32)

#         return x, y, label

#     def _make_indices(self, labels, normal_only):
#         """
#         生成可用窗口起点。

#         对于训练/验证:
#             normal_only=True
#             要求 input window + future patch 全部 label==0。

#         对于测试:
#             normal_only=False
#             保留所有窗口。
#         """
#         labels = labels.reshape(-1)

#         max_start = len(labels) - self.win_size - self.pred_len
#         if max_start < 0:
#             return []

#         indices = []

#         for start in range(0, max_start + 1, self.step):
#             s_begin = start
#             s_end = s_begin + self.win_size

#             r_begin = s_end
#             r_end = r_begin + self.pred_len

#             if normal_only:
#                 window_label = labels[s_begin:r_end]
#                 if window_label.max() > 0:
#                     continue

#             indices.append(start)

#         return indices

#     def _select_data(self):
#         if self.flag == "train":
#             return self.train_x, self.train_y, self.train_labels, self.train_indices
#         elif self.flag == "val":
#             return self.val_x, self.val_y, self.val_labels, self.val_indices
#         elif self.flag == "test":
#             return self.test_x, self.test_y, self.test_labels, self.test_indices
#         else:
#             raise ValueError(f"unknown flag: {self.flag}")

#     def __len__(self):
#         _, _, _, indices = self._select_data()
#         return len(indices)

#     def __getitem__(self, index):
#         data_x, data_y, labels, indices = self._select_data()

#         start = indices[index]

#         s_begin = start
#         s_end = s_begin + self.win_size

#         r_begin = s_end
#         r_end = r_begin + self.pred_len

#         seq_x = data_x[s_begin:s_end]
#         seq_y = data_y[r_begin:r_end]
#         label_y = labels[r_begin:r_end].reshape(-1)

#         return np.float32(seq_x), np.float32(seq_y), np.float32(label_y), np.array([start], dtype=np.int64)
    