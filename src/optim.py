from typing import Dict, Iterator, List, Tuple

from torch.nn.parameter import Parameter
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


def build_adam_RLROP(
    parameters: Iterator[Parameter],
    initial_learning_rate: float = 0.01,
    factor: float = 0.1,
    patience: int = 5,
    cooldown: int = 2,
    use_train_loss: bool = False,
) -> Tuple[List[Optimizer], List[Dict[str, object]]]:
    optimizer = Adam(parameters, lr=initial_learning_rate)

    # Define ReduceLROnPlateau scheduler
    if use_train_loss:
        monitor = "train_loss"
    else:
        monitor = "val_loss"
    scheduler = {
        "scheduler": ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            cooldown=cooldown,
        ),
        # Metric to monitor
        "monitor": monitor,
        # Check interval: 'epoch' or 'step'
        "interval": "epoch",
        # Check learning rate every n epochs
        "frequency": 1,
        "name": "lr",
    }

    return [optimizer], [scheduler]
