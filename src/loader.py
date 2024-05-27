import gc
import os
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import TensorDataset

CHANNEL_COUNT = 62
MAX_DATA_LEN = 11041
MAX_SHAPE = (CHANNEL_COUNT, MAX_DATA_LEN)
POSSIBLE_CHANNELS = [
    "FP1",
    "FPZ",
    "FP2",
    "AF3",
    "AF4",
    "F7",
    "F5",
    "F3",
    "F1",
    "FZ",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "T7",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "P7",
    "P5",
    "P3",
    "P1",
    "PZ",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO7",
    "PO5",
    "PO3",
    "POZ",
    "PO4",
    "PO6",
    "PO8",
    "CB1",
    "O1",
    "OZ",
    "O2",
    "CB2",
]


class Task(IntEnum):
    LEFT_RIGHT = 1
    UP_DOWN = 2
    TWO_DIM = 3


class Target(IntEnum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


@dataclass
class DataSet:
    data: np.ndarray
    labels: np.ndarray
    online_accuracy: float
    forced_online_accuracy: float
    class_balance: Tuple[int, ...]
    ptp_refused_percent: float


def make_dataset(dataset: DataSet) -> TensorDataset:
    data = torch.tensor(dataset.data, dtype=torch.float32)
    labels = torch.tensor(dataset.labels, dtype=torch.uint8)
    return TensorDataset(data, labels)


def load(
    directory_path: str,
    filter_task: Optional[Task] = None,
    time_frame: Optional[Tuple[int, int]] = None,
    filter_channels: List[str] = POSSIBLE_CHANNELS,
    ptp_thresh: Optional[int] = 130,
) -> DataSet:
    shape = MAX_SHAPE
    if filter_channels:
        shape = (len(filter_channels), shape[1])
    if time_frame:
        shape = (shape[0], time_frame[1] - time_frame[0])
    out_data = []
    out_labels = []
    success_count = 0
    forced_success_count = 0
    class_balance = [0, 0, 0, 0]
    total_trials = 0
    refused_trials = 0
    for file in os.listdir(directory_path):
        if not file.endswith(".mat"):
            continue
        file_path = directory_path + "/" + file
        print(f"Loading file {file_path}...")
        file_data = loadmat(file_path, simplify_cells="True")["BCI"]
        channel_indices = np.array(
            [
                np.where(chan == file_data["chaninfo"]["label"])
                for chan in filter_channels
            ]
        ).squeeze()

        for data, trial_data in zip(
            file_data["data"], file_data["TrialData"], strict=True
        ):
            task = Task(trial_data["tasknumber"])
            if filter_task is not None and task != filter_task:
                continue
            data = data[channel_indices]
            if time_frame is not None:
                data = data[:, time_frame[0] : min(time_frame[1], data.shape[1])]
            if ptp_thresh:
                total_trials += 1
                ptp_maxamp = np.max(np.ptp(data, axis=1))
                if ptp_maxamp > ptp_thresh:
                    refused_trials += 1
                    continue
            if data.shape != shape:
                pad_height = shape[0] - data.shape[0]
                pad_width = shape[1] - data.shape[1]
                data = np.pad(data, pad_width=((0, pad_height), (0, pad_width)))
            if trial_data["result"] == 1:
                success_count += 1
            if trial_data["forcedresult"]:
                forced_success_count += 1
            out_data.append(data)
            label = trial_data["targetnumber"] - 1
            class_balance[label] += 1
            out_labels.append(label)
    gc.collect()
    print(out_data[0].shape)
    if ptp_thresh:
        refused_percent = (refused_trials / total_trials) * 100
    else:
        refused_percent = 0
    return DataSet(
        data=np.array(out_data, dtype=np.float32),
        labels=np.array(out_labels, dtype=np.uint8),
        online_accuracy=success_count / len(out_data),
        forced_online_accuracy=forced_success_count / len(out_data),
        class_balance=tuple(class_balance),
        ptp_refused_percent=refused_percent,
    )


if __name__ == "__main__":
    from sys import argv

    used_channels = [chan for chan in POSSIBLE_CHANNELS if "C" in chan]
    ds = load(
        argv[1], time_frame=(2000, 6000), filter_channels=used_channels, ptp_thresh=100
    )
    print(
        f"Online accuracy: {ds.online_accuracy * 100:.2f}%\tForced online accuracy: {ds.forced_online_accuracy * 100:.2f}%"
    )
    print(f"Class balance: {ds.class_balance}")
    print(f"Refused by ptp threshold: {ds.ptp_refused_percent:.2f}%")
    tds = make_dataset(ds)
    torch.save(tds, argv[2])
