import gc
import os
from dataclasses import dataclass
from enum import IntEnum
from sys import argv
from typing import Dict, List, Literal, Optional, Set, Tuple

import numpy as np
import torch
from lightning_fabric.plugins.environments.slurm import re
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
class LoadDomain:
    task: Optional[Task]
    time_frame: Optional[Tuple[int, int]]
    channels: List[str]
    
    def __str__(self):
        chan_string = ""
        for i, chan in enumerate(self.channels):
            chan_string += f"{chan},".ljust(5)
            if i % 10 == 9:
                chan_string += "\n"
        return f"""Task: {self.task.name.replace("_", " ") if self.task else 'ALL'}
Time Frame: {self.time_frame if self.time_frame else 'FULL'}
{len(self.channels)} Channels:
{chan_string}"""


@dataclass
class SubjectSpec:
    gender: Optional[Literal["M", "F"]] = None
    age_range: Optional[Tuple[int, int]] = None
    handedness: Optional[Literal["R", "L"]] = None
    mbsr: Optional[bool] = None
    meditation_hours_range: Optional[Tuple[float, float]] = None
    instrument: Optional[List[Literal["Y", "U", "N"]]] = None
    athlete: Optional[List[Literal["Y", "U", "N"]]] = None
    handsport: Optional[List[Literal["Y", "U", "N"]]] = None
    hobby: Optional[List[Literal["Y", "U", "N"]]] = None

    def passes(self, subject_metadata: Dict) -> bool:
        if self.gender and subject_metadata["gender"] != self.gender:
            return False
        if self.age_range and not (self.age_range[0] <= subject_metadata["age"] < self.age_range[1]):
            return False
        if self.handedness and subject_metadata["handedness"] != self.handedness:
            return False
        if self.mbsr is not None and bool(subject_metadata["MBSRsubject"]) != self.mbsr:
            return False
        if self.meditation_hours_range and not (
            self.meditation_hours_range[0] <= subject_metadata["meditationpractice"] < self.meditation_hours_range[1]
        ):
            return False
        if self.instrument and subject_metadata["instrument"] not in self.instrument:
            return False
        if self.athlete and subject_metadata["athlete"] not in self.athlete:
            return False
        if self.handsport and subject_metadata["handsport"] not in self.handsport:
            return False
        if self.hobby and subject_metadata["hobby"] not in self.hobby:
            return False
        return True

    def __str__(self):
        return f"""Gender: {self.gender if self.gender else 'ANY'}
Age Range: {self.age_range if self.age_range else 'ANY'}
Handedness: {self.handedness if self.handedness else 'ANY'}
MBSR attended: {self.mbsr if self.mbsr else 'IRRELEVANT'}
Meditation Time (Hours): {self.meditation_hours_range if self.meditation_hours_range else 'ANY'}
Instrument Player: {self.instrument if self.instrument else 'IRRELEVANT'}
Athlete: {self.athlete if self.athlete else 'IRRELEVANT'}
Handsport Player: {self.handsport if self.handsport else 'IRRELEVANT'}
Hobby: {self.hobby if self.hobby else 'IRRELEVANT'}"""


@dataclass
class DataSet:
    ds: TensorDataset
    item_shape: Size
    online_accuracy: float
    forced_online_accuracy: float
    class_balance: Tuple[int, ...]
    ptp_refused: float
    ptp_thresh: Optional[int]
    subjects: Set[int]
    domain: LoadDomain
    subject_spec: Optional[SubjectSpec] = None

    def print_stats(self):
        print()
        print("----------------------  DATASET SIZE  --------------------")
        print(f"Item shape:\t{self.item_shape}")
        print(f"Total items:\t{len(self.ds)}")
        print()

        print("------------------------  DOMAIN  ------------------------")
        print(self.domain)
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

        print("----------------  SUBJECT SPECIFICATION  -----------------")
        if self.subject_spec:
            print(self.subject_spec)
        else:
            print("No subject specification applied.")
        print()

        print("-----------------------  SUBJECTS  -----------------------")
        sub_string = ""
        for i, chan in enumerate(self.subjects):
            sub_string += f"{chan},".ljust(4)
            if i % 10 == 9:
                sub_string += "\n"
        print(f"{len(self.subjects)} subjects:")
        print(sub_string)
        print()


def extract_integers(filename: str) -> Tuple[int | None, int | None]:
    pattern = r"S(\d+)_Session_(\d+)\.mat"
    
    match = re.search(pattern, filename)
    
    if match:
        subject = int(match.group(1))
        session = int(match.group(2))
        return subject, session
    else:
        return None, None

class Loader:
    def __init__(
        self,
        domain: LoadDomain = LoadDomain(task=None, time_frame=(2000, 6000), channels=POSSIBLE_CHANNELS),
        subject_spec: Optional[SubjectSpec] = None,
        ptp_thresh: Optional[int] = 130,
        ) -> None:
        self.domain = domain
        self.subject_spec = subject_spec
        self.filter_task = domain.task
        self.time_frame = domain.time_frame
        self.filter_channels = domain.channels
        self.ptp_thresh = ptp_thresh

        self.shape = MAX_SHAPE
        self.shape = (len(self.filter_channels), self.shape[1])
        if self.time_frame:
            self.shape = (self.shape[0], self.time_frame[1] - self.time_frame[0])
        self.out_data = []
        self.out_labels = []
        self.success_count = 0
        self.forced_success_count = 0
        self.class_balance = [0, 0, 0, 0]
        self.total_trials = 0
        self.refused_trials = 0
        self.subjects: Set[int] = set()

    def load_dir(
            self,
        directory_path: str,
    ) -> DataSet:
        files = os.listdir(directory_path)
        file_count = len(files)
        for i, file in enumerate(files, start=1):
            file_path = directory_path + "/" + file
            print(f"Processing file ({i}/{file_count}) {file_path}...")
            if not file.endswith(".mat"):
                print(f"Skipping {file_path} (not a .mat file).")
                continue
            try:
                file_data = loadmat(file_path, simplify_cells="True")["BCI"]
                if self.subject_spec and not self.subject_spec.passes(file_data["metadata"]):
                    print(f"Skipping {file_path} (subject does not fit specification).")
                    continue
                subject, _ = extract_integers(file)
                if subject:
                    self.subjects.add(subject)
                self.process_data(file_data)
            except Exception as e:
                print(f"Invalid file: {file_path}, received error: {e}")
            gc.collect()
        if self.ptp_thresh:
            refused = self.refused_trials / self.total_trials
        else:
            refused = 0
        data = torch.tensor(np.array(self.out_data, dtype=np.float32), dtype=torch.float32)
        labels = torch.tensor(np.array(self.out_labels, dtype=np.uint8), dtype=torch.uint8)
        ds = TensorDataset(data, labels)
        return DataSet(
            ds=ds,
            ptp_thresh=self.ptp_thresh,
            item_shape=ds[0][0].shape,
            online_accuracy=self.success_count / len(self.out_data),
            forced_online_accuracy=self.forced_success_count / len(self.out_data),
            class_balance=tuple(self.class_balance),
            ptp_refused=refused,
            subjects=self.subjects,
            domain=self.domain,
            subject_spec=self.subject_spec,
        )

    def process_data(
            self,
            file_data: Dict,
    ):
            channel_indices = np.array(
                [
                    np.where(chan == file_data["chaninfo"]["label"])
                    for chan in self.filter_channels
                ]
            ).squeeze()

            for data, trial_data in zip(
                file_data["data"], file_data["TrialData"], strict=True
            ):
                task = Task(trial_data["tasknumber"])
                if self.filter_task is not None and task != self.filter_task:
                    continue
                data = data[channel_indices]
                if self.time_frame is not None:
                    data = data[:, self.time_frame[0] : min(self.time_frame[1], data.shape[1])]
                if self.ptp_thresh:
                    self.total_trials += 1
                    ptp_maxamp = np.max(np.ptp(data, axis=1))
                    if ptp_maxamp > self.ptp_thresh:
                        self.refused_trials += 1
                        continue
                if data.shape[1] != self.shape[1]:
                    pad_width = self.shape[1] - data.shape[1]
                    # print(f"Padding occurs! Adding {pad_width} samples to {data.shape[1]} samples.")
                    # print(f"Original: {data[0]}")
                    data = np.pad(data, pad_width=((0, 0), (0, pad_width)), mode="reflect")
                    # print(f"Padded: {data[0]}")
                if trial_data["result"] == 1:
                    self.success_count += 1
                if trial_data["forcedresult"]:
                    self.forced_success_count += 1
                self.out_data.append(data)
                label = trial_data["targetnumber"] - 1
                self.class_balance[label] += 1
                self.out_labels.append(label)

if __name__ == "__main__":
    used_channels = [chan for chan in POSSIBLE_CHANNELS if "C" in chan]
    domain = LoadDomain(task=None, time_frame=(2000, 6000), channels=used_channels)
    subject_spec = SubjectSpec(gender="F", age_range=(18, 28), handedness="R")
    loader = Loader(domain=domain, subject_spec=subject_spec)
    ds = loader.load_dir(argv[1])
    ds.print_stats()
    torch.save(ds, argv[2])

