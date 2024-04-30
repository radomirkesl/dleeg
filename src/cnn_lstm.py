import pytorch_lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torchmetrics import Accuracy


class CNN_LSTM_1D(L.LightningModule):

    def __init__(
        self,
        data_shape,
        in_channels,
        conv1_kernel = 5,
        conv2_kernel = 7,
        conv1_filters = 64,
        conv2_filters = 128,
        pool_kernel = 4,
        hidden_size=64,
        feature_count=4,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                kernel_size=conv1_kernel,
                out_channels=conv1_filters,
                padding='same',
            ),
            nn.BatchNorm1d(conv1_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel, stride=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=conv1_filters,
                kernel_size=conv2_kernel,
                out_channels=conv2_filters,
                padding='same'
            ),
            nn.BatchNorm1d(conv2_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel, stride=1),
        )
        self.lstm = nn.LSTM(
                input_size=self.feature_count_after_convs(data_shape),
                hidden_size=hidden_size,
        )
        self.full = nn.Linear(
            in_features=hidden_size, out_features=feature_count
        )
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = torch.flatten(x, 1)
        x, _ = self.lstm(x)

        x = self.full(x)

        return x

    def feature_count_after_convs(self, data_shape):
        x = torch.zeros(data_shape)
        x = self.conv1(x)
        x = self.conv2(x)

        return x.numel()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        # print(f'Outputs shape: {outputs.shape}\tLabels shape: {labels.shape}')
        # print(f'Outputs: {outputs}\nLabels: {labels}')
        loss = self.loss(outputs, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        val_loss = self.loss(outputs, labels)
        self.log('val_loss', val_loss)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        test_loss = self.loss(outputs, labels)
        self.accuracy(outputs, torch.argmax(labels, dim=1))
        self.log('test_acc', self.accuracy, on_step=False, on_epoch=True)
        self.log('test_loss', test_loss)


class CNN_LSTM_2D(L.LightningModule):

    def __init__(
        self,
        data_shape,
        conv_kernel_size,
        pool_kernel_size,
        pool_stride,
        hidden_size=8,
        feature_count=4,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=conv_kernel_size,
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=conv_kernel_size
        )
        self.pool = nn.MaxPool2d(pool_kernel_size, pool_stride)
        # self.input_to_hidden = nn.Linear(in_features=self.feature_count_after_pools(data_shape), out_features=hidden_size)
        self.hidden_to_output = nn.Linear(
            in_features=hidden_size, out_features=feature_count
        )
        self.loss = nn.CrossEntropyLoss()
        self.lstm = nn.LSTM(
            input_size=self.feature_count_after_pools(data_shape),
            hidden_size=hidden_size,
        )

    def forward(self, x):
        # print('Forwarding...')
        # print(f'Input shape: {x.shape}')
        # Convolve once
        x = self.conv1(x)
        # print(f'After first conv: {x.shape}')
        x = F.relu(x)
        x = self.pool(x)
        # print(f'After first pooling: {x.shape}')

        # Convolve again
        x = self.conv2(x)
        # print(f'After second conv: {x.shape}')
        x = F.relu(x)
        x = self.pool(x)
        # print(f'After second pooling: {x.shape}')

        x = torch.flatten(x, 1)
        # print(f'After flattening: {x.shape}')

        # x = self.input_to_hidden(x)
        # print(f'After input to hidden: {x.shape}')
        # x = F.relu(x)
        x, _ = self.lstm(x)
        x = self.hidden_to_output(x)
        # print(f'After hidden to output: {x.shape}')

        return x

    def feature_count_after_pools(self, data_shape):
        x = torch.zeros(data_shape)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        count = x.numel()
        return count

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        # print(f'Outputs shape: {outputs.shape}\tLabels shape: {labels.shape}')
        # print(f'Outputs: {outputs}\nLabels: {labels}')
        loss = self.loss(outputs, labels)
        return loss



if __name__ == "__main__":
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from torch.utils.data import random_split

    from loader import *

    L.seed_everything(seed=42, workers=True)

    used_channels = [chan for chan in POSSIBLE_CHANNELS if 'C' in chan]
    print(f'Channel count: {len(used_channels)}')
    # ds = load(
    #     "../data",
    #     time_frame=(2000, 6000),
    #     filter_task=None,
    #     filter_channels=used_channels,
    # )
    # tds = make_dataset(ds)
    tds: TensorDataset = torch.load('sub01_ses05-11_chansC.ds')
    sets = random_split(tds, [0.64, 0.16, 0.2])
    train, val, test = tuple(DataLoader(s, num_workers=3, batch_size=16) for s in sets)

    model = CNN_LSTM_1D(
        tds[0][0].shape,
        in_channels=len(used_channels)
    )

    trainer = L.Trainer(max_epochs=10, callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=3)])
    trainer.fit(model, train_dataloaders=train, val_dataloaders=val)
    trainer.test(model, test)
    trainer.save_checkpoint("cnn_lstm.ckpt")

