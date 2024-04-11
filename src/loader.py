from typing_extensions import deprecated
from scipy.io import loadmat
from dataclasses import dataclass
import numpy as np
from enum import IntEnum
from typing import Optional, Tuple
import os
from torch.utils.data import TensorDataset, DataLoader, random_split

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

def make_loader(dataset: DataSet, workers = 0) -> DataLoader:
    tds = TensorDataset(dataset.data, dataset.labels)
    return DataLoader(tds, num_workers=workers)

def make_split_loaders(dataset: DataSet, lengths: Tuple[float, ...], workers = 0) -> Tuple[DataLoader, ...]:
    tds = TensorDataset(dataset.data, dataset.labels)
    sets = random_split(tds, lengths)
    return tuple(DataLoader(s, num_workers=workers) for s in sets)

# @dataclass
# class DataSet:
#     data: np.ndarray
#     labels: np.ndarray
#     online_accuracy: float
#     forced_online_accuracy: float

def load_tensor(folder_path: str, filter_task: Optional[Task] = None, time_frame: Optional[Tuple[int, int]] = None) -> DataSet:
    if time_frame is None:
        shape = DATA_SHAPE
    else:
        shape = (CHANNEL_COUNT, time_frame[1] - time_frame[0])
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
        for data, trial_data in zip(file_data['data'], file_data['TrialData'], strict = True):
            task = Task(trial_data['tasknumber'])
            if filter_task is not None and task != filter_task:
                continue
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
            data = torch.tensor(np.array(out_data, dtype = np.float32), dtype = torch.float32),
            labels = torch.tensor(np.array(out_labels, dtype = np.float32), dtype = torch.float32),
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
        file_path = folder_path + '/' + file
        print(f'Loading file {file_path}...')
        file_data = loadmat(file_path, simplify_cells = 'True')['BCI']
        for data, trial_data in zip(file_data['data'], file_data['TrialData'], strict = True):
            task = Task(trial_data['tasknumber'])
            if filter_task is not None and task != filter_task:
                continue
            if data.shape != DATA_SHAPE:
                pad_height = DATA_SHAPE[0] - data.shape[0]
                pad_width = DATA_SHAPE[1] - data.shape[1]
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
            data = torch.from_numpy(np.array(out_data, dtype = np.float32)),
            labels = torch.from_numpy(np.array(out_labels, dtype = np.float32)),
            online_accuracy = success_count / len(out_data),
            forced_online_accuracy = forced_success_count / len(out_data))

@deprecated('Use load_tensor or load_from_numpy instead')
def load_numpy_2(folder_path: str, filter_task: Optional[Task] = None) -> DataSet:
    out_data = []
    out_labels = []
    success_count = 0
    forced_success_count = 0
    for file in os.listdir(folder_path):
        if not file.endswith('.mat'):
            continue
        file_data = loadmat(folder_path + '/' + file, simplify_cells = 'True')['BCI']
        data = file_data['data']
        trial_data = file_data['TrialData']
        if filter_task is not None:
            valid_indices = np.array([trial['tasknumber'] == filter_task for trial in trial_data])
            data = data[valid_indices]
            trial_data = filter(lambda trial: trial['tasknumber'] == filter_task, trial_data)
        for d, td in zip(data, trial_data, strict = True):
            if d.shape != DATA_SHAPE:
                pad_height = DATA_SHAPE[0] - d.shape[0]
                pad_width = DATA_SHAPE[1] - d.shape[1]
                d = np.pad(d, pad_width = ((0, pad_height), (0, pad_width)))
            if td['result'] == 1:
                success_count += 1
            forced_success_count += td['forcedresult']
            out_data.append(d)
            out_labels.append(td['targetnumber'])
    return DataSet(
            data = torch.tensor(np.array(out_data, dtype = np.float32), dtype = torch.float32),
            labels = torch.tensor(np.array(out_labels, dtype = np.uint8), dtype = torch.uint8),
            online_accuracy = success_count / len(out_data),
            forced_online_accuracy = forced_success_count / len(out_data)
            )

@deprecated('Use load_tensor or load_from_numpy instead')
def load_numpy_1(folder_path: str, filter_task: Optional[Task] = None) -> DataSet:
    out_data: np.ndarray = np.empty(shape = (0, DATA_SHAPE[0], DATA_SHAPE[1]), dtype = np.float32)
    out_labels: np.ndarray = np.empty(shape = (0), dtype = np.uint8)
    success_count = 0
    forced_success_count = 0
    for file in os.listdir(folder_path):
        if not file.endswith('.mat'):
            continue
        file_data = loadmat(folder_path + '/' + file, simplify_cells = 'True')['BCI']
        data = file_data['data']
        trial_data = file_data['TrialData']
        labels = np.array([trial['targetnumber'] for trial in trial_data], dtype = np.uint8)
        if filter_task is not None:
            valid_indices = np.array([trial['tasknumber'] == filter_task for trial in trial_data])
            data = data[valid_indices]
            labels = labels[valid_indices]
        success_count += np.sum([1 if trial['result'] == 1 else 0 for trial in trial_data])
        forced_success_count += np.sum([trial['forcedresult'] for trial in trial_data])
        padded = np.zeros((len(data), DATA_SHAPE[0], DATA_SHAPE[1]), dtype = np.float32)
        for i, arr in enumerate(data):
            padded[i, :arr.shape[0], :arr.shape[1]] = arr
        out_data = np.concatenate((out_data, padded))
        out_labels = np.concatenate((out_labels, labels))
    return DataSet(
            data = torch.tensor(out_data, dtype = torch.float32),
            labels = torch.tensor(out_labels, dtype = torch.uint8),
            online_accuracy = success_count / out_data.shape[0],
            forced_online_accuracy = forced_success_count / out_data.shape[0]
            )

if __name__ == "__main__":
    ds = load_tensor('../data', time_frame = (2000, 6000))
    print(ds.data.shape)
    print(ds.data[0].shape)
    print(ds.labels.shape)
    print(ds.data[0])
    data2 = ds.data.permute(0, 2, 1)
    print(data2.shape)
    print(data2[0])

