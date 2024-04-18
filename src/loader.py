from scipy.io import loadmat
from dataclasses import dataclass
import numpy as np
from enum import IntEnum
from typing import Optional, Tuple, List
import os
from torch.utils.data import TensorDataset, DataLoader, random_split

import torch

CHANNEL_COUNT = 62
MAX_DATA_LEN = 11041
MAX_SHAPE = (CHANNEL_COUNT, MAX_DATA_LEN)
DESIRED_CHANNELS = ['CZ', 'C3', 'C4']

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

def make_loader(dataset: DataSet, workers = 0) -> DataLoader:
    tds = TensorDataset(dataset.data, dataset.labels)
    return DataLoader(tds, num_workers=workers)

def make_split_loaders(dataset: DataSet, lengths: Tuple[float, ...], workers = 0, batch_size=1) -> Tuple[DataLoader, ...]:
    tds = TensorDataset(dataset.data, dataset.labels)
    sets = random_split(tds, lengths)
    return tuple(DataLoader(s, num_workers=workers, batch_size=batch_size) for s in sets)

# @dataclass
# class DataSet:
#     data: np.ndarray
#     labels: np.ndarray
#     online_accuracy: float
#     forced_online_accuracy: float

def load(folder_path: str, filter_task: Optional[Task] = None, time_frame: Optional[Tuple[int, int]] = None, filter_channels: Optional[List[str]] = None) -> DataSet:
    shape = MAX_SHAPE
    if filter_channels:
        shape = (len(filter_channels), shape[1])
    if time_frame:
        shape = (shape[0], time_frame[1] - time_frame[0])
    out_data = []
    out_labels = []
    success_count = 0
    forced_success_count = 0
    for file in os.listdir(folder_path):
        if not file.endswith('.mat'):
            continue
        file_path = folder_path + '/' + file
        print(f'Loading file {file_path}...')
        file_data = loadmat(file_path, simplify_cells = 'True')['BCI']
        channel_indices = None
        if filter_channels:
            channel_indices = np.array([np.where(chan == file_data['chaninfo']['label']) for chan in filter_channels]).squeeze()
        for data, trial_data in zip(file_data['data'], file_data['TrialData'], strict = True):
            task = Task(trial_data['tasknumber'])
            if filter_task is not None and task != filter_task:
                continue
            if channel_indices is not None:
                data = data[channel_indices]
            if time_frame is not None:
                data = data[:, time_frame[0]:min(time_frame[1], data.shape[1])]
            if data.shape != shape:
                pad_height = shape[0] - data.shape[0]
                pad_width = shape[1] - data.shape[1]
                data = np.pad(data, pad_width = ((0, pad_height), (0, pad_width)))
            if trial_data['result'] == 1:
                success_count += 1
            if trial_data['forcedresult']:
                forced_success_count += 1
            out_data.append(data)
            label = [0.0, 0.0, 0.0, 0.0]
            label[trial_data['targetnumber'] - 1] = 1.0
            out_labels.append(label)
    return DataSet(
            data = torch.tensor(np.array(out_data, dtype = np.float32), dtype = torch.float32).unsqueeze(dim = 1),
            labels = torch.tensor(np.array(out_labels, dtype = np.float32), dtype = torch.float32),
            online_accuracy = success_count / len(out_data),
            forced_online_accuracy = forced_success_count / len(out_data)
            )


if __name__ == "__main__":
    ds = load('../data', time_frame = (2000, 6000))
    print(ds.data.shape)
    print(ds.data[0].shape)
    print(ds.labels.shape)
    print(ds.data[0])

