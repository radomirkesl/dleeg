import gc
import os
from dataclasses import dataclass
from enum import IntEnum
from sys import argv
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.io import loadmat
from torch import Size
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
    ds: TensorDataset
    item_shape: Size
    used_channels: List[str]
    online_accuracy: float
    forced_online_accuracy: float
    class_balance: Tuple[int, ...]
    ptp_refused: float
    used_task: Optional[Task]
    time_frame: Optional[Tuple[int, int]]
    ptp_thresh: Optional[int]

    def print_stats(self):
        print("----------------------  DATASET SIZE  --------------------")
        print(f"Item shape:\t{self.item_shape}")
        print(f"Total items:\t{len(self.ds)}")
        print()

        print("-----------------  TIME-SPACE SELECTION  -----------------")
        print(f"{len(self.used_channels)} channels used: {self.used_channels}")
        print(f"Time frame: {self.time_frame}")
        print()

        print("--------------------  ONLINE RESULTS  --------------------")
        print(f"Online accuracy:\t{self.online_accuracy * 100:.2f}%")
        print(f"Forced online accuracy:\t{self.forced_online_accuracy * 100:.2f}%")
        print()

        print("---------------  ARTIFACT REMOVAL EFFECTS  ---------------")
        print(f"Point to point threshold:\t{self.ptp_thresh}")
        print(f"Refused by ptp threshold:\t{self.ptp_refused * 100:.2f}%")
        print(f"Class balance:\t{self.class_balance}")
        print()

        print("-------------------------  TASK  -------------------------")
        print(f"Task used:\t{self.used_task if self.used_task else 'All'}")
        print()


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
    if ptp_thresh:
        refused_percent = refused_trials / total_trials
    else:
        refused_percent = 0
    data = torch.tensor(np.array(out_data, dtype=np.float32), dtype=torch.float32)
    labels = torch.tensor(np.array(out_labels, dtype=np.uint8), dtype=torch.uint8)
    ds = TensorDataset(data, labels)
    return DataSet(
        ds=ds,
        time_frame=time_frame,
        ptp_thresh=ptp_thresh,
        used_task=filter_task,
        item_shape=ds[0][0].shape,
        used_channels=filter_channels,
        online_accuracy=success_count / len(out_data),
        forced_online_accuracy=forced_success_count / len(out_data),
        class_balance=tuple(class_balance),
        ptp_refused=refused_percent,
    )


if __name__ == "__main__":
    used_channels = [chan for chan in POSSIBLE_CHANNELS if "C" in chan]
    ds = load(
        argv[1],
        time_frame=(2000, 6000),
        filter_channels=used_channels,
    )
    ds.print_stats()
    torch.save(ds, argv[2])

