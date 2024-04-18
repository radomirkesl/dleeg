import torch
import pytorch_lightning as L
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

class CNN_LSTM(L.LightningModule):

    def __init__(self, data_shape, conv_kernel_size, pool_kernel_size, pool_stride, hidden_size = 8, feature_count = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=conv_kernel_size,
        )
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=conv_kernel_size
        )
        self.pool = nn.MaxPool2d(pool_kernel_size, pool_stride)
        self.input_to_hidden = nn.Linear(in_features=self.feature_count_after_pools(data_shape), out_features=hidden_size)
        self.hidden_to_output = nn.Linear(in_features=hidden_size, out_features=feature_count)
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
        return Adam(self.parameters(), lr = 0.01)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        # print(f'Outputs shape: {outputs.shape}\tLabels shape: {labels.shape}')
        # print(f'Outputs: {outputs}\nLabels: {labels}')
        loss = self.loss(outputs, labels)
        return loss


if __name__ == "__main__":

    from loader import load, make_split_loaders, Task

    L.seed_everything(seed=42, workers=True)

    ds = load('../data', time_frame = (2000, 6000), filter_task=Task.LEFT_RIGHT, filter_channels=['CZ', 'C3', 'C4'])
    train, test = make_split_loaders(ds, (0.8, 0.2), batch_size=1, workers=3)

    model = CNN_LSTM(ds.data[0].shape, conv_kernel_size=(1, 5), pool_kernel_size=(1,4), pool_stride=1, hidden_size=8)

    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model, train_dataloaders=train)
    correct = 0
    for batch_num, (trial, label) in enumerate(test):
        pred = model(trial)
        pred_label = torch.round(torch.softmax(pred, dim=1), decimals=2)
        print(f'Predicted label:\t{pred_label}\tTrue label:\t{label}')
        closest = torch.argmax(pred_label)
        if closest == torch.argmax(label):
            correct += 1
    print(f'Accuracy: {correct / len(test)}')
    trainer.save_checkpoint('cnn_lstm.ckpt')

