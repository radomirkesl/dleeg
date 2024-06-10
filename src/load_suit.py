import json
from pathlib import Path
from sys import argv

import torch

from loader import Loader, SubjectSpec


def load_and_process(
    experiment_name: str, subject_spec: SubjectSpec, inputs_dir: Path, outputs_dir: Path
):
    loader = Loader(subject_spec=subject_spec)
    print("------------------------------------")
    print(f"Loading data for {experiment_name}")
    print("------------------------------------")
    outputs_dir /= experiment_name
    outputs_dir.mkdir(exist_ok=True)
    try:
        ds, stats = loader.load_dir(inputs_dir)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    with open(outputs_dir / "data_stats.json", "w") as f:
        json.dump(stats.to_dict_hr(), f, indent=4)
    torch.save(ds, outputs_dir / "data.ds")


if __name__ == "__main__":
    if len(argv) != 3:
        print("Usage: python load_suit.py <inputs_dir> <outputs_dir>")
        exit(1)
    inputs_dir = Path(argv[1])
    if not inputs_dir.exists():
        print(f"Path {inputs_dir} does not exist.")
        exit(1)
    outputs_dir = Path(argv[2])
    if not outputs_dir.exists():
        print(f"Path {outputs_dir} does not exist.")
        exit(1)

    load_and_process(
        "all_18-30", SubjectSpec(age_range=(18, 30)), inputs_dir, outputs_dir
    )
    load_and_process(
        "all_30-50", SubjectSpec(age_range=(30, 50)), inputs_dir, outputs_dir
    )
    load_and_process(
        "all_50-64", SubjectSpec(age_range=(50, 64)), inputs_dir, outputs_dir
    )
