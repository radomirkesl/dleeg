from datetime import timedelta
from time import time
from typing import List, Optional

import pytorch_lightning as L
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split


class Runner:
    def __init__(
        self,
        model: L.LightningModule,
        data: TensorDataset,
        save_path: Optional[str] = None,
        workers: int = 3,
        batch_size: int = 32,
        pin_memory: bool = True,
        seed: int = 42,
        max_epochs: int = 200,
        patience: int = 20,
        set_split: List = [0.64, 0.16, 0.2],
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
                EarlyStopping(monitor="val_loss", mode="min", patience=patience)
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
        model: L.LightningModule,
        data: TensorDataset,
        save_path: Optional[str] = None,
        workers: int = 3,
        batch_size: int = 32,
        pin_memory: bool = True,
        seed: int = 42,
        max_epochs: int = 100,
        patience: int = 10,
        num_folds: int = 5,
        set_split: List = [0.8, 0.2],
    ) -> None:

        L.seed_everything(seed=seed, workers=True)

        self.model: L.LightningModule = model
        self.save_path: Optional[str] = save_path
        self.workers: int = workers
        self.batch_size: int = batch_size
        self.pin_memory: bool = pin_memory
        self.max_epochs: int = max_epochs
        self.patience: int = patience

        self.train_data, test_data = random_split(data, set_split)
        kfold = KFold(n_splits=num_folds, shuffle=True)
        self.folds = list(kfold.split(self.train_data))
        self.test = DataLoader(
            test_data,
            num_workers=workers,
            batch_size=batch_size,
            shuffle=False,
            persistent_workers=True,
            pin_memory=pin_memory,
        )

    def run(self):
        tick = time()
        for fold, (train_idx, val_idx) in enumerate(self.folds):
            print(f"FOLD {fold}")
            print("--------------------------------")

            train_subset = Subset(self.train_data, train_idx)
            val_subset = Subset(self.train_data, val_idx)

            train_loader = DataLoader(
                train_subset,
                num_workers=self.workers,
                batch_size=self.batch_size,
                shuffle=True,
                persistent_workers=True,
                pin_memory=self.pin_memory,
            )
            val_loader = DataLoader(
                val_subset,
                num_workers=self.workers,
                batch_size=self.batch_size,
                shuffle=False,
                persistent_workers=True,
                pin_memory=self.pin_memory,
            )

            # checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

            self.trainer = L.Trainer(
                max_epochs=self.max_epochs,
                callbacks=[
                    EarlyStopping(
                        monitor="val_loss", patience=self.patience, mode="min"
                    ),
                    # checkpoint_callback,
                ],
            )

            self.trainer.fit(
                self.model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
            )

        self.trainer.test(self.model, self.test)
        if self.save_path:
            self.trainer.save_checkpoint(self.save_path)
        tock = time()
        print(f"Elapsed time: {timedelta(seconds=tock - tick)}")
