from scipy.io import loadmat
from dataclasses import dataclass
import numpy as np
from enum import IntEnum
from typing import Optional
import os

import torch

CHANNEL_COUNT = 62
MAX_DATA_LEN = 11041
DATA_SHAPE = (CHANNEL_COUNT, MAX_DATA_LEN)

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
    data: torch.Tensor
    labels: torch.Tensor
    online_accuracy: float
    forced_online_accuracy: float

# @dataclass
# class DataSet:
#     data: np.ndarray
#     labels: np.ndarray
#     online_accuracy: float
#     forced_online_accuracy: float

def load_tensor(folder_path: str, filter_task: Optional[Task] = None) -> DataSet:
    out_data = []
    out_labels = []
    success_count = 0
    forced_success_count = 0
    for file in os.listdir(folder_path):
        if not file.endswith('.mat'):
            continue
        file_data = loadmat(folder_path + '/' + file, simplify_cells = 'True')['BCI']
        # TODO: Try doing this with numpy
        for data, trial_data in zip(file_data['data'], file_data['TrialData'], strict = True):
            task = Task(trial_data['tasknumber'])
            if filter_task is not None and task != filter_task:
                continue
            if data.shape != DATA_SHAPE:
                pad_height = DATA_SHAPE[0] - data.shape[0]
                pad_width = DATA_SHAPE[1] - data.shape[1]
                data = np.pad(data, pad_width = ((0, pad_height), (0, pad_width)))
            np.ndarray.resize
            if trial_data['result'] == 1:
                success_count += 1
            if trial_data['forcedresult']:
                forced_success_count += 1
            out_data.append(data)
            out_labels.append(trial_data['targetnumber'])
    return DataSet(
            data = torch.tensor(np.array(out_data, dtype = np.float32), dtype = torch.float32),
            labels = torch.tensor(np.array(out_labels, dtype = np.uint8), dtype = torch.uint8),
            online_accuracy = success_count / len(out_data),
            forced_online_accuracy = forced_success_count / len(out_data)
            )


def load_from_numpy(folder_path: str, filter_task: Optional[Task] = None) -> DataSet:
    out_data = []
    out_labels = []
    success_count = 0
    forced_success_count = 0
    for file in os.listdir(folder_path):
        if not file.endswith('.mat'):
            continue
        file_data = loadmat(folder_path + '/' + file, simplify_cells = 'True')['BCI']
        # TODO: Try doing this with numpy
        for data, trial_data in zip(file_data['data'], file_data['TrialData'], strict = True):
            task = Task(trial_data['tasknumber'])
            if filter_task is not None and task != filter_task:
                continue
            if data.shape != DATA_SHAPE:
                pad_height = DATA_SHAPE[0] - data.shape[0]
                pad_width = DATA_SHAPE[1] - data.shape[1]
                data = np.pad(data, pad_width = ((0, pad_height), (0, pad_width)))
            np.ndarray.resize
            if trial_data['result'] == 1:
                success_count += 1
            if trial_data['forcedresult']:
                forced_success_count += 1
            out_data.append(data)
            out_labels.append(trial_data['targetnumber'])
    return DataSet(
            data = torch.from_numpy(np.array(out_data, dtype = np.float32)),
            labels = torch.from_numpy(np.array(out_labels, dtype = np.uint8)),
            online_accuracy = success_count / len(out_data),
            forced_online_accuracy = forced_success_count / len(out_data))

