from sys import argv

import torch

from cnn_lstm import CNN_LSTM
from loader import *
from run import Runner

if __name__ == "__main__":

    ds: TensorDataset = torch.load(argv[1])
    model = CNN_LSTM(data_shape=ds.tensors[0][0].shape)
    if len(argv) > 2:
        model_save = argv[2]
    else:
        model_save = None
    runner = Runner(model=model, data=ds, save_path=model_save)
    runner.run()
