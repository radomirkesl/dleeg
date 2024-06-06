from sys import argv

import torch

from cnn_lstm import CNN_LSTM
from loader import *
from run import Runner

if __name__ == "__main__":

    ds: DataSet = torch.load(argv[1])
    ds.print_stats()
    model = CNN_LSTM((1, *ds.item_shape), in_channels=ds.item_shape[0], lstm_layers=1)
    if len(argv) > 2:
        model_save = argv[2]
    else:
        model_save = None
    runner = Runner(model=model, data=ds.ds, save_path=model_save, max_epochs=1)
    runner.run()
