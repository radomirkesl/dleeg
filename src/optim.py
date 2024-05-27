from typing import Dict, Iterator, List, Tuple

import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import ReduceLROnPlateau


def build_adam_RLROP(
    parameters: Iterator[Parameter],
    initial_learning_rate: float = 0.1,
    factor: float = 0.1,
    patience: int = 5,
    cooldown: int = 5,
    verbose=True,
) -> Tuple[List[optim.Optimizer], List[Dict[str, object]]]:
    optimizer = optim.Adam(parameters, lr=initial_learning_rate)

    # Define ReduceLROnPlateau scheduler
    scheduler = {
        "scheduler": ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            cooldown=cooldown,
            verbose=verbose,
        ),
        # Metric to monitor
        "monitor": "val_loss",
        # Check interval: 'epoch' or 'step'
        "interval": "epoch",
        # Check learning rate every n epochs
        "frequency": 1,
    }

    return [optimizer], [scheduler]
