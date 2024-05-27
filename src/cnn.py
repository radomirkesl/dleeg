import pytorch_lightning as L
import torch
from torch import nn
from torch.optim import Adam
from torchmetrics import Accuracy

from optim import build_adam_RLROP


class CNN(L.LightningModule):

    def __init__(
        self,
        data_shape,
        in_channels,
        conv1_kernel=64,
        conv2_kernel=16,
        conv1_filters=8,
        conv2_filters=16,
        pool_kernel=4,
        feature_count=4,
        conv_depth=2,
        dropout_rate=0.5,
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
        self.full = nn.Linear(
            in_features=self.feature_count_after_convs(data_shape),
            out_features=feature_count,
        )
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = torch.flatten(x, 1)

        x = self.full(x)

        return x

    def feature_count_after_convs(self, data_shape):
        x = torch.zeros(data_shape)
        x = self.conv1(x)
        x = self.conv2(x)

        return x.numel()

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
        test_loss = self.loss(outputs, labels)
        self.accuracy(outputs, labels)
        self.log("test_acc", self.accuracy, on_step=False, on_epoch=True)
        self.log("test_loss", test_loss)


if __name__ == "__main__":
    from sys import argv

    from torch.utils.data import TensorDataset

    from run import Runner

    tds: TensorDataset = torch.load(argv[1])
    data_shape = tds[0][0].shape
    print(f"Data shape: {data_shape}")
    model = CNN((1, *data_shape), in_channels=data_shape[0])
    if len(argv) > 2:
        model_save = argv[2]
    else:
        model_save = None
    runner = Runner(model=model, data=tds, save_path=model_save)
    runner.run()
