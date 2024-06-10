from typing import Dict

import pytorch_lightning as L
import torch
from torch import nn
from torchmetrics.classification import MulticlassConfusionMatrix

from metrics import CLASS_NUM, build_classwise_metrics, build_general_metrics
from optim import build_adam_RLROP


class CNN_LSTM(L.LightningModule):

    def __init__(
        self,
        in_channels=4,
        conv1_kernel=64,
        conv2_kernel=16,
        conv1_filters=8,
        conv2_filters=16,
        pool_kernel=4,
        feature_count=4,
        conv_depth=2,
        dropout_rate=0.5,
        hidden_size=128,
        lstm_layers=3,
    ):
        super().__init__()
        conv1_out = conv_depth * conv1_filters
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                kernel_size=conv1_kernel,
                out_channels=conv1_filters,
                padding="same",
            ),
            nn.BatchNorm1d(conv1_filters),
            # Depthwise conv
            nn.Conv1d(
                in_channels=conv1_filters,
                groups=conv1_filters,
                kernel_size=conv1_kernel,
                out_channels=conv1_out,
                padding="same",
            ),
            nn.BatchNorm1d(conv1_out),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=pool_kernel, stride=1),
            nn.Dropout(dropout_rate),
        )
        self.conv2 = nn.Sequential(
            # Depthwise separable conv
            nn.Conv1d(
                in_channels=conv1_out,
                kernel_size=conv2_kernel,
                out_channels=conv1_out,
                groups=conv1_out,
                padding="same",
            ),
            nn.Conv1d(
                in_channels=conv1_out,
                out_channels=conv2_filters,
                kernel_size=1,
            ),
            nn.BatchNorm1d(conv2_filters),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=pool_kernel, stride=1),
            nn.Dropout(dropout_rate),
        )
        self.lstm = nn.LSTM(
            input_size=conv2_filters,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=lstm_layers,
        )
        self.full = nn.Linear(
            in_features=hidden_size,
            out_features=feature_count,
        )
        self.loss = nn.CrossEntropyLoss()

        self.general_metrics = build_general_metrics()
        self.classwise_metrics = build_classwise_metrics()
        self.cm = MulticlassConfusionMatrix(num_classes=CLASS_NUM, normalize="true")
        self.saved_metrics: Dict

        self.save_hyperparameters()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.full(x[:, -1, :])

        return x

    def configure_optimizers(self):
        return build_adam_RLROP(self.parameters())

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.loss(outputs, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        val_loss = self.loss(outputs, labels)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        metrics = self.general_metrics(outputs, labels)
        self.log_dict(self.general_metrics, on_step=False, on_epoch=True)
        self.saved_metrics = metrics
        self.saved_metrics.update(self.classwise_metrics(outputs, labels))
        self.saved_metrics["confusion_matrix"] = self.cm(outputs, labels)


if __name__ == "__main__":
    from sys import argv

    from torch.utils.data import TensorDataset

    from run import Runner

    ds: TensorDataset = torch.load(argv[1])
    model = CNN_LSTM(lstm_layers=1)
    if len(argv) > 2:
        model_save = argv[2]
    else:
        model_save = None
    runner = Runner(model=model, data=ds, save_path=model_save)
    runner.run()
