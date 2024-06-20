import json
from datetime import timedelta
from pathlib import Path
from sys import argv
from time import time
from typing import Type

import pytorch_lightning as L
import torch
from torch.utils.data import TensorDataset

from cnn_lstm import CNN_LSTM
from run import KFoldRunner


class TrainSuite:
    def __init__(
        self,
        top_dir: Path,
        model_type: Type[L.LightningModule],
        save_name: str,
        overwrite: bool = False,
        dry_run: bool = False,
        **kwargs,
    ):
        self.top_dir = top_dir
        self.model_type = model_type
        self.save_name = save_name
        self.overwrite = overwrite
        self.top_dir.mkdir(exist_ok=True)
        self.dry_run = dry_run
        self.kwargs = kwargs

    def run(self):
        for experiment_path in self.top_dir.iterdir():
            if not experiment_path.is_dir():
                continue
            experiment_name = experiment_path.name
            data_path = experiment_path / "data.ds"
            results_path = experiment_path / (self.save_name + "_results.json")
            model_path = experiment_path / (self.save_name + ".ckpt")
            if not data_path.exists():
                print(f"Skipping {experiment_name} as data.ds does not exist")
                continue
            if not self.overwrite and results_path.exists():
                print(
                    f"Skipping {experiment_name} as {results_path.name} exists, to retrain use CLSuite(overwrite=True)"
                )
                continue
            print("----------------------------------------")
            print(f"Training {self.model_type.__name__} for {experiment_name}")
            print("----------------------------------------")
            tick = time()
            try:
                ds: TensorDataset = torch.load(data_path)
                if self.dry_run:
                    runner = KFoldRunner(
                        model_type=self.model_type,
                        data=ds,
                        save_path=model_path,
                        max_epochs=1,
                        num_folds=3,
                        data_shape=ds.tensors[0][0].shape,
                        **self.kwargs,
                    )
                else:
                    runner = KFoldRunner(
                        model_type=self.model_type,
                        data=ds,
                        save_path=model_path,
                        data_shape=ds.tensors[0][0].shape,
                        **self.kwargs,
                    )
                runner.run()
                stats = runner.get_aggregated_metrics()
                with open(results_path, "w") as f:
                    json.dump(stats, f, indent=4)
            except Exception as e:
                print(
                    f"Failed to train {self.model_type.__name__} for {experiment_name}"
                )
                print(e)
            tock = time()
            print("--------------------------------------------")
            print(
                f"Training {self.model_type.__name__} for {experiment_name} took {timedelta(seconds=tock - tick)}"
            )
            print("--------------------------------------------")
            if self.dry_run:
                break


if __name__ == "__main__":
    if not len(argv) == 2:
        print("Usage: python train_suite.py <path to directory with experiments>")
        exit(1)
    top_dir = Path(argv[1])
    if not top_dir.exists():
        print(f"{top_dir} does not exist")
        exit(1)
    if not top_dir.is_dir():
        print(f"{top_dir} is not a directory")
        exit(1)

    suite = TrainSuite(
        top_dir=top_dir,
        model_type=CNN_LSTM,
        save_name="cnn_lstm",
    )
    suite.run()
