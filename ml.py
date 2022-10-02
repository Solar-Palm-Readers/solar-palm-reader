"""
Based on a tutorial:
https://github.com/gabrielloye/GRU_Prediction
"""

import os
import time
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class GRUNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, drop_prob: float = 0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.tensor, h: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size: int) -> torch.nn.Parameter:
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


class LSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, drop_prob: float = 0.2):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.tensor, h: torch.tensor) -> torch.tensor:
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size: int) -> Tuple[torch.nn.Parameter, torch.nn.Parameter]:
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden


def train(train_loader: DataLoader, learn_rate: float, batch_size: int, hidden_dim: int = 256, num_epochs: int = 5,
          model_type: str = "GRU") -> nn.Module:
    # Setting common hyperparameters
    input_dim = next(iter(train_loader))[0].shape[2]
    output_dim = 1
    n_layers = 2
    # Instantiating the models
    if model_type == "GRU":
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    else:
        model = LSTMNet(input_dim, hidden_dim, output_dim, n_layers)
    model.to(device)

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    print("Starting Training of {} model".format(model_type))
    epoch_times = []
    # Start training loop
    for epoch in range(1, num_epochs + 1):
        start_time = time.clock()
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0
        for x, label in train_loader:
            counter += 1
            if model_type == "GRU":
                h = h.data
            else:
                h = tuple([e.data for e in h])
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if counter % 200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter,
                                                                                           len(train_loader),
                                                                                           avg_loss / counter))
        current_time = time.clock()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, num_epochs, avg_loss / len(train_loader)))
        print("Time Elapsed for Epoch: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    return model


def evaluate(model: nn.Module, test_x: Dict[str, np.ndarray], test_y: Dict[str, np.ndarray],
             label_scalers: Dict[str, MinMaxScaler]) -> Tuple[List[torch.Tensor], List[torch.Tensor], float]:
    model.eval()
    outputs = []
    targets = []
    start_time = time.clock()
    for i in test_x.keys():
        inp = torch.from_numpy(np.array(test_x[i]))
        labs = torch.from_numpy(np.array(test_y[i]))
        h = model.init_hidden(inp.shape[0])
        out, h = model(inp.to(device).float(), h)
        outputs.append(label_scalers[i].inverse_transform(out.cpu().detach().numpy()).reshape((-1,)))
        targets.append(label_scalers[i].inverse_transform(labs.numpy()).reshape((-1,)))
    print("Evaluation Time: {}".format(str(time.clock() - start_time)))

    sMAPE = 0
    for i in range(len(outputs)):
        sMAPE += np.mean(abs(outputs[i] - targets[i]) / (targets[i] + outputs[i]) / 2) / len(outputs)
    print("sMAPE: {}%".format(sMAPE * 100))

    return outputs, targets, sMAPE


def main() -> None:
    # Define data root directory
    data_dir = "./data/"
    print(os.listdir(data_dir))

    pd.read_csv(data_dir + 'AEP_hourly.csv').head()

    label_scalers = {}

    train_x = []
    train_y = []
    test_x = {}
    test_y = {}

    for file in tqdm(os.listdir(data_dir)):
        # Skipping the files we're not using
        if file[-4:] != ".csv" or file == "pjm_hourly_est.csv":
            continue

        # Store csv file in a Pandas DataFrame
        df = pd.read_csv(data_dir + file, parse_dates=[0])

        # Scaling the input data
        sc = MinMaxScaler()
        label_sc = MinMaxScaler()
        data = sc.fit_transform(df.values)

        # Obtaining the Scale for the labels so that output can be re-scaled to actual value during evaluation
        label_sc.fit(df.iloc[:, 0].values.reshape((-1, 1)))
        label_scalers[file] = label_sc

        # Define lookback period and split inputs/labels
        lookback = 100
        inputs = np.zeros((len(data) - lookback, lookback, df.shape[1]))
        labels = np.zeros(len(data) - lookback)

        for i in range(lookback, len(data)):
            inputs[i - lookback] = data[i - lookback:i]
            labels[i - lookback] = data[i, 0]
        inputs = inputs.reshape((-1, lookback, df.shape[1]))
        labels = labels.reshape((-1, 1))

        # Split data into train/test portions and combining all data from different files into a single array
        test_portion = int(0.1 * len(inputs))
        if len(train_x) == 0:
            train_x = inputs[:-test_portion]
            train_y = labels[:-test_portion]
        else:
            train_x = np.concatenate((train_x, inputs[:-test_portion]))
            train_y = np.concatenate((train_y, labels[:-test_portion]))
        test_x[file] = (inputs[-test_portion:])
        test_y[file] = (labels[-test_portion:])

    print(train_x.shape)

    batch_size = 1024

    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

    lr = 0.001
    gru_model = train(train_loader, lr, batch_size, model_type="GRU")

    lstm_model = train(train_loader, lr, batch_size, model_type="LSTM")

    evaluate(gru_model, test_x, test_y, label_scalers)

    evaluate(lstm_model, test_x, test_y, label_scalers)


if __name__ == '__main__':
    main()
