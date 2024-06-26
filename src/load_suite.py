import json
from datetime import timedelta
from pathlib import Path
from sys import argv
from time import time
from typing import Optional

import torch

from loader import Loader, SubjectSpec


class LoadSuite:
    def __init__(self, inputs_dir: Path, outputs_dir: Path, overwrite: bool = False):
        self.inputs_dir = inputs_dir
        self.outputs_dir = outputs_dir
        self.overwrite = overwrite

    def load_and_process(
        self,
        experiment_name: str,
        subject_spec: Optional[SubjectSpec] = None,
        loader: Optional[Loader] = None,
    ):
        if not loader:
            if not subject_spec:
                print(
                    f"No subject specification nor loader provided for {experiment_name}, no action taken."
                )
                return
            loader = Loader(subject_spec=subject_spec)
        experiment_output = self.outputs_dir / experiment_name
        experiment_output.mkdir(exist_ok=True)
        data_path = experiment_output / "data.ds"
        if not self.overwrite and data_path.exists():
            print(
                f"Data already exists for {experiment_name}, skipping. Use LoadSuite(overwrite=True) to overwrite."
            )
            return
        print("------------------------------------")
        print(f"Loading data for {experiment_name}")
        print("------------------------------------")
        tick = time()
        try:
            ds, stats = loader.load_dir(self.inputs_dir)
        except Exception as e:
            print(f"Error loading data: {e}")
            return
        with open(experiment_output / "data_stats.json", "w") as f:
            json.dump(stats.to_dict_hr(), f, indent=4)
        torch.save(ds, data_path)
        tock = time()
        print("--------------------------------------------")
        print(f"Loading {experiment_name} took {timedelta(seconds=tock - tick)}")
        print("--------------------------------------------")


if __name__ == "__main__":
    if len(argv) != 3:
        print("Usage: python load_suite.py <inputs_dir> <outputs_dir>")
        exit(1)
    inputs_dir = Path(argv[1])
    if not inputs_dir.exists():
        print(f"Path {inputs_dir} does not exist.")
        exit(1)
    outputs_dir = Path(argv[2])
    if not outputs_dir.exists():
        print(f"Path {outputs_dir} does not exist.")
        exit(1)
    if not inputs_dir.is_dir():
        print(f"{inputs_dir} is not a directory.")
        exit(1)
    if not outputs_dir.is_dir():
        print(f"{outputs_dir} is not a directory.")
        exit(1)

    suite = LoadSuite(inputs_dir, outputs_dir)

    suite.load_and_process(
        "all_0-30",
        SubjectSpec(age_range=(0, 30)),
    )
    suite.load_and_process(
        "all_30-50",
        SubjectSpec(age_range=(30, 50)),
    )
    suite.load_and_process(
        "all_50-100",
        SubjectSpec(age_range=(50, 100)),
    )
    suite.load_and_process(
        "athlete",
        SubjectSpec(athlete=["Y"]),
    )
    # TODO: Maybe delete
    suite.load_and_process(
        "sub1",
        loader=Loader(subject_nums=[1]),
    )
