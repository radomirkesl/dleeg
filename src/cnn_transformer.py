import pytorch_lightning as L
import torch
from torch import nn
from torchmetrics import Accuracy

from optim import build_adam_RLROP


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class CNNTransformer(L.LightningModule):

    def __init__(
        self,
        data_shape,
        conv1_kernel=64,
        conv2_kernel=16,
        conv1_filters=8,
        conv2_filters=16,
        pool_kernel=4,
        num_classes=4,
        conv_depth=2,
        dropout_rate=0.5,
        transformer_heads=8,
        transformer_layers=8,
    ):
        super().__init__()
        conv1_out = conv_depth * conv1_filters
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=data_shape[0],
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

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(
            d_model=conv2_filters,
            max_len=data_shape[1],
        )

        # Transformer Layers
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=conv2_filters,
            nhead=transformer_heads,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.transformer_layer,
            num_layers=transformer_layers,
        )

        self.full = nn.Linear(
            in_features=self.feature_count_after_convs(data_shape),
            out_features=num_classes,
        )

        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # Flatten and prepare for Transformer
        x = x.permute(0, 2, 1)  # (seq_length, batch_size, features)
        x = self.positional_encoding(x)

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # Flatten and Fully Connected Layer
        x = torch.flatten(x, start_dim=1)
        x = self.full(x)

        return x

    def feature_count_after_convs(self, data_shape):
        x = torch.zeros((1, data_shape[0], data_shape[1]))
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

    ds: TensorDataset = torch.load(argv[1])
    model = CNNTransformer(data_shape=ds.tensors[0][0].shape)
    if len(argv) > 2:
        model_save = argv[2]
    else:
        model_save = None
    runner = Runner(model=model, data=ds, save_path=model_save)
    runner.run()
