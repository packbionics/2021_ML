import torch
from torch import nn
from torch.nn import Module
from torch.cuda import init
from torch.utils.data import Dataset


class RNNDataset(Dataset):
    def __init__(self, x, y, window_size, offset):
        self.x = x
        self.y = y
        self.window = window_size
        self.offset = offset

    def __getitem__(self, index):
        # if index <= len()
        _x = self.x[index:index+self.window]
        _y = self.y[index + self.window + self.offset]  # 1 y for every (window) x
        return _x, _y

    def __len__(self):
        return len(self.x) - self.window - self.offset


class RNNModel(Module):
    def __init__(self, batch_size, sequence_size, hidden_size, rnn_layers) -> None:
        super().__init__()
        self.h_0 = torch.zeros(rnn_layers, batch_size, 9)

        # Model layers
        self.rnn_input = nn.LSTM(input_size=9, hidden_size=hidden_size, num_layers=rnn_layers,
                                 dropout=0.2, batch_first=True)

        # out --> (direction * num_layers, batch, hidden) (hidden state) SQUEEZE
        self.linear1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 16)
        self.fc_out = nn.Linear(16, 1)

    def forward(self, x):
        out, (hidden, cell) = self.rnn_input(x)
        x = self.linear1(hidden)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        return self.fc_out(x)

class NNDataset(Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.x = x
        self.y = y
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)
    
class NNModel(Module):
    def __init__(self, features_in) -> None:
        super().__init__()
        print('FEATS', features_in)
        self.linear1 = nn.Linear(9, 16)
        self.linear2 = nn.Linear(16, 8)
        self.dense_out = nn.Linear(8, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dense_out(x)
        return x
    
        
class ConvDataset(Dataset):
    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = X
        self.y = y
    
    def __getitem__(self, index):
        _x = self.x[index]

class ConvModel(Module):
    # Assuming channels = features, 
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.linear1 = torch.nn.Linear(out_channels, 8)
        self.relu = torch.nn.ReLU()
        self.fc_out = torch.nn.Linear(8, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.linear1(x)
        x = self.fc_out(x)
        return x