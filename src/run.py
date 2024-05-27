from copy import deepcopy
from typing import List, Optional

import numpy as np
import pytorch_lightning as L
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
from torchvision import datasets, transforms


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
        self.save_path: Optional[str] = save_path

        L.seed_everything(seed=seed, workers=True)

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
        self.trainer.fit(
            self.model,
            train_dataloaders=self.train,
            val_dataloaders=self.val,
        )
        self.trainer.test(self.model, self.test)
        if self.save_path:
            self.trainer.save_checkpoint(self.save_path)


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
        max_epochs: int = 200,
        patience: int = 20,
        num_folds: int = 5,
    ) -> None:
        self.model: L.LightningModule = model
        self.data: TensorDataset = data
        self.save_path: Optional[str] = save_path
        self.workers: int = workers
        self.batch_size: int = batch_size
        self.pin_memory: bool = pin_memory
        self.max_epochs: int = max_epochs
        self.patience: int = patience
        self.num_folds: int = num_folds

        L.seed_everything(seed=seed, workers=True)

        kfold = KFold(n_splits=num_folds, shuffle=True)
        self.folds = list(kfold.split(data))

    def run(self):
        results = []
        for fold, (train_idx, val_idx) in enumerate(self.folds):
            print(f'FOLD {fold}')
            print('--------------------------------')

            train_subset = Subset(self.data, train_idx)
            val_subset = Subset(self.data, val_idx)

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

            model = deepcopy(self.model)

            # Define callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, mode='min')
            checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min')

            self.trainer = L.Trainer(
                    max_epochs=self.max_epochs,
                    callbacks=[
                        early_stopping,
                        checkpoint_callback,
                        ],
                    )

            self.trainer.fit(
                    model,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                    )

            # Save results for this fold
            if checkpoint_callback.best_model_score:
                vl = checkpoint_callback.best_model_score.item()
            else:
                vl = None
            results.append({
                'fold': fold,
                'checkpoint_path': checkpoint_callback.best_model_path,
                'val_loss': vl,
                })

        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {self.num_folds} FOLDS')
        print('--------------------------------')
        for result in results:
            print(f"Fold {result['fold']} | Best Val Loss: {result['val_loss']} | Checkpoint Path: {result['checkpoint_path']}")

        val_losses = [result['val_loss'] for result in results]
        mean_val_loss = np.mean(val_losses)
        std_val_loss = np.std(val_losses)

        print(f'Average Validation Loss: {mean_val_loss}')
        print(f'Standard Deviation of Validation Loss: {std_val_loss}')

