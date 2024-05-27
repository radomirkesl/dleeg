from sys import argv

import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, random_split

from cnn_lstm import CNN_LSTM
from loader import *

if __name__ == "__main__":

    L.seed_everything(seed=42, workers=True)

    used_channels = [chan for chan in POSSIBLE_CHANNELS if "C" in chan]
    print(f"Channel count: {len(used_channels)}")

    tds: TensorDataset = torch.load(argv[1])
    data_shape = tds[0][0].shape
    print(f"Data shape: {data_shape}")
    sets = random_split(tds, [0.64, 0.16, 0.2])
    train = DataLoader(
        sets[0],
        num_workers=3,
        batch_size=32,
        shuffle=True,
        persistent_workers=True,
        pin_memory=False,
    )
    val, test = tuple(
        DataLoader(
            s,
            num_workers=3,
            batch_size=32,
            shuffle=False,
            persistent_workers=True,
            pin_memory=False,
        )
        for s in sets[1:]
    )

    model = CNN_LSTM((1, *data_shape), in_channels=data_shape[0], lstm_layers=1)

    trainer = L.Trainer(
        max_epochs=200,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=20)],
    )
    trainer.fit(model, train_dataloaders=train, val_dataloaders=val)
    trainer.test(model, test)
    if len(argv) > 2:
        trainer.save_checkpoint(argv[2])
