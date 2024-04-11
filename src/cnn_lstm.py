import torch
import pytorch_lightning as L
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

class CNN_LSTM(L.LightningModule):

    def __init__(self, data_shape, conv_kernel_size, pool_kernel_size, pool_stride, cnn_hidden_size = 8, feature_count = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=conv_kernel_size
        )
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=conv_kernel_size
        )
        self.pool = nn.MaxPool2d(pool_kernel_size, pool_stride)
        self.input_to_hidden = nn.Linear(in_features=2400, out_features=cnn_hidden_size)
        self.hidden_to_output = nn.Linear(in_features=cnn_hidden_size, out_features=feature_count)
        self.loss = nn.CrossEntropyLoss()
        self.lstm = nn.LSTM(
            input_size=feature_count,
            hidden_size=feature_count,
        )

    def forward(self, x):
        # Convolve once
        self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Convolve again
        print(x.shape)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)

        # x = self.input_to_hidden(x)
        # x = F.relu(x)
        # x = self.hidden_to_output(x)

        # Shove through LSTM
        x, _ = self.lstm(x)
        return x

    def feature_count_after_pool(self, data_shape):
        x = torch.zeros((1, data_shape[0], data_shape[1]))
        print(x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        print(x.shape)
        return x.shape[1]

    def configure_optimizers(self):
        return Adam(self.parameters(), lr = 0.1)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.loss(outputs, labels)
        return loss


class CNN_LSTM_whatever(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers):
        super(CNN_LSTM_whatever, self).__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.c1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size = 2, stride = 1) # Add a 1D CNN layer
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers
        )
        self.linear = nn.Linear(in_features=n_hidden, out_features=1)
    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len-1, self.n_hidden),
            torch.zeros(self.n_layers, self.seq_len-1, self.n_hidden)
        )
    def forward(self, sequences):
        sequences = self.c1(sequences.view(len(sequences), 1, -1))
        lstm_out, self.hidden = self.lstm(
            sequences.view(len(sequences), self.seq_len-1, -1),
            self.hidden
        )
        last_time_step = lstm_out.view(self.seq_len-1, len(sequences), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred

class CNN_LSTM_other(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CNN_LSTM_other, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        #cnn takes input of shape (batch_size, channels, seq_len)
        x = x.permute(0, 1)
        out = self.cnn(x)
        # lstm takes input of shape (batch_size, seq_len, input_size)
        out = out.permute(0, 1)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out

    def reset_hidden_state(self):
            self.hidden = (
                torch.zeros(self.num_layers, self.input_size-1, self.hidden_size),
                torch.zeros(self.num_layers, self.input_size-1, self.hidden_size)
            )


if __name__ == "__main__":

    from loader import load_tensor, make_split_loaders, CHANNEL_COUNT

    L.seed_everything(seed=42, workers=True)
    model = CNN_LSTM((CHANNEL_COUNT, 4000), conv_kernel_size=(2, 100), pool_kernel_size=2, pool_stride=2)
    ds = load_tensor('../data', time_frame = (2000, 6000))
    print(ds.data.shape)
    train, test = make_split_loaders(ds, (0.8, 0.2))
    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model, train_dataloaders=train)
    correct = 0
    for batch_num, (trial, label) in enumerate(test):
        pred = model(trial)
        pred_label = torch.round(torch.softmax(pred, dim=1), decimals=2)
        print(f'Predicted label:\t{pred_label}\tTrue label:\t{label}')
        closest = torch.argmin(torch.abs(pred_label - label))
        if closest == torch.argmax(label):
            print('Correct!')
            correct += 1
    print(f'Accuracy: {correct / len(test)}')

