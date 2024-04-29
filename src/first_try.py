import time
from sys import argv
from typing import Callable

import torch

import loader
from trainer import train_model_unvalidated


def experiment_zero(loader_func: Callable):
    data_set = loader_func(argv[1], filter_task=loader.Task.TWO_DIM)
    print(f"Online success rate: {data_set.online_accuracy * 100:.2f}%")
    print(f"Online forced success rate: {data_set.forced_online_accuracy * 100:.2f}%")
    train_len = int(0.8 * len(data_set.data))
    train_data = data_set.data[:train_len]
    print("Train data type:", train_data.dtype)
    print("Train data shape:", train_data.shape)
    train_labels = data_set.labels[:train_len]
    print("Train labels type:", train_labels.dtype)
    print("Train labels shape:", train_labels.shape)
    test_data = data_set.data[train_len:]
    print("Test data type:", test_data.dtype)
    print("Test data shape:", test_data.shape)
    test_labels = data_set.labels[train_len:]
    print("Test labels type:", test_labels.dtype)
    print("Test labels shape:", test_labels.shape)

    model = CNN_LSTM_whatever(n_features=4, n_hidden=4, n_layers=1, seq_len=62)
    # model = CNN_LSTM_other(
    #             num_classes=4,
    #             hidden_size=4,
    #             input_size=62,
    #             num_layers=1
    #         )
    model, _ = train_model_unvalidated(
        model=model,
        train_data=train_data,
        train_labels=train_labels,
        num_epochs=100,
    )
    with torch.no_grad():
        preds = []
        for _ in range(len(test_data)):
            model.reset_hidden_state()
            y_test_pred = model(torch.unsqueeze(test_data[_], 0))
            pred = torch.flatten(y_test_pred).item()
            preds.append(pred)
        correct = 0
        for pred, label in zip(preds, test_labels, strict=True):
            if pred == label:
                correct += 1
        print(f"Accuracy: {correct / len(test_data) * 100:.2f}%")


if __name__ == "__main__":
    print("Underlying ndarray:")
    tick = time.time()
    experiment_zero(loader.load_from_numpy)
    tock = time.time()
    print(f"Time elapsed: {tock - tick:.2f} seconds")

    print("\n---\n")

    print("Direct Tensor:")
    tick = time.time()
    experiment_zero(loader.load_tensor)
    tock = time.time()
    print(f"Time elapsed: {tock - tick:.2f} seconds")

    # print('\n---\n')
    #
    # print('Direct Tensor calculated using NumPy:')
    # tick = time.time()
    # experiment_zero(loader.load_numpy_1)
    # tock = time.time()
    # print(f'Time elapsed: {tock - tick:.2f} seconds')
    #
    # print('\n---\n')
    #
    # print('Direct Tensor calculated using NumPy v2:')
    # tick = time.time()
    # experiment_zero(loader.load_numpy_2)
    # tock = time.time()
    # print(f'Time elapsed: {tock - tick:.2f} seconds')
