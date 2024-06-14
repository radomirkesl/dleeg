from datetime import timedelta
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Tuple, Type

import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split


class Runner:
    def __init__(
        self,
        model: L.LightningModule,
        data: TensorDataset,
        save_path: Optional[str] = None,
        workers: int = 11,
        batch_size: int = 64,
        pin_memory: bool = True,
        seed: int = 42,
        max_epochs: int = 200,
        patience: int = 20,
        set_split: Tuple[float, float, float] = (0.64, 0.16, 0.2),
    ) -> None:
        L.seed_everything(seed=seed, workers=True)

        self.save_path: Optional[str] = save_path

        self.model: L.LightningModule = model
        sets = random_split(data, set_split)
        self.train = DataLoader(
            sets[0],
            num_workers=workers,
            batch_size=batch_size,
            shuffle=True,
            persistent_workers=True,
            pin_memory=pin_memory,
        )
        self.val, self.test = tuple(
            DataLoader(
                s,
                num_workers=workers,
                batch_size=batch_size,
                shuffle=False,
                persistent_workers=True,
                pin_memory=pin_memory,
            )
            for s in sets[1:]
        )

        self.trainer = L.Trainer(
            max_epochs=max_epochs,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=patience),
                LearningRateMonitor(logging_interval="epoch"),
            ],
        )

    def run(self):
        tick = time()
        self.trainer.fit(
            self.model,
            train_dataloaders=self.train,
            val_dataloaders=self.val,
        )
        self.trainer.test(self.model, self.test)
        if self.save_path:
            self.trainer.save_checkpoint(self.save_path)
        tock = time()
        print(f"Elapsed time: {timedelta(seconds=tock - tick)}")


class KFoldRunner:
    def __init__(
        self,
        model_type: Type[L.LightningModule],
        data: TensorDataset,
        save_path: Optional[Path] = None,
        workers: int = 11,
        batch_size: int = 64,
        pin_memory: bool = True,
        seed: int = 42,
        max_epochs: int = 100,
        patience: int = 15,
        num_folds: int = 5,
        **kwargs,
    ) -> None:

        L.seed_everything(seed=seed, workers=True)

        self._model_type: Type[L.LightningModule] = model_type
        self._data: TensorDataset = data
        self._save_path: Optional[Path] = save_path
        self._workers: int = workers
        self._batch_size: int = batch_size
        self._pin_memory: bool = pin_memory
        self._max_epochs: int = max_epochs
        self._patience: int = patience
        self._num_folds: int = num_folds
        self._model_params = kwargs

        kfold = KFold(n_splits=num_folds, shuffle=True)
        self._folds = list(kfold.split(self._data))
        self._metrics: List[Dict] = []

    def run(self):
        tick = time()
        for fold, (train_idx, val_idx) in enumerate(self._folds):
            print(f"FOLD {fold}")
            print("--------------------------------")

            train_subset = Subset(self._data, train_idx)
            val_subset = Subset(self._data, val_idx)

            train_loader = DataLoader(
                train_subset,
                num_workers=self._workers,
                batch_size=self._batch_size,
                shuffle=True,
                persistent_workers=True,
                pin_memory=self._pin_memory,
            )
            val_loader = DataLoader(
                val_subset,
                num_workers=self._workers,
                batch_size=self._batch_size,
                shuffle=False,
                persistent_workers=True,
                pin_memory=self._pin_memory,
            )
            model = self._model_type(**self._model_params)
            trainer = L.Trainer(
                    max_epochs=self._max_epochs,
                    callbacks=[
                        EarlyStopping(monitor="val_loss", patience=self._patience, mode="min"),
                        LearningRateMonitor(logging_interval="epoch"),
                        ],
                    )

            trainer.fit(
                model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )
            trainer.test(model, val_loader)
            self._metrics.append(model.saved_metrics)

        print("FULL SET TRAINING")
        print("--------------------------------")
        full_loader = DataLoader(
            self._data,
            num_workers=self._workers,
            batch_size=self._batch_size,
            shuffle=True,
            persistent_workers=True,
            pin_memory=self._pin_memory,
        )
        model = self._model_type(**self._model_params, rlrop_use_train_loss=True)
        trainer = L.Trainer(
                max_epochs=self._max_epochs,
                callbacks=[
                    EarlyStopping(monitor="train_loss", patience=self._patience, mode="min"),
                    LearningRateMonitor(logging_interval="epoch"),
                    ],
                )

        trainer.fit(
            model,
            train_dataloaders=full_loader,
        )

        if self._save_path:
            trainer.save_checkpoint(self._save_path)
        tock = time()
        print(f"Elapsed time: {timedelta(seconds=tock - tick)}")

    def get_aggregated_metrics(self) -> Dict[str, Dict[str, float]]:
        metrics = {}
        for m in self._metrics:
            for k, v in m.items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)
        metrics = {k: torch.std_mean(torch.tensor(v), dim=0) for k, v in metrics.items()}
        return {k: {"mean": v[1].item(), "std": v[0].item()} for k, v in metrics.items()}


