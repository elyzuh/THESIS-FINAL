import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()

        self.use_cuda = args.cuda
        self.m = data.m
        self.w = args.window
        self.hidR = args.hidRNN
        self.dropout = args.dropout

        self.lstm = nn.LSTM(
            input_size=self.m,
            hidden_size=self.hidR,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout
        )

        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(self.hidR, self.hidR)
        self.fc2 = nn.Linear(self.hidR, self.m)
        self.relu = nn.ReLU()

        self.output = None
        if args.output_fun == 'sigmoid':
            self.output = torch.sigmoid
        elif args.output_fun == 'tanh':
            self.output = torch.tanh

    def forward(self, x):
        # x: batch x window x m

        lstm_out, (h_n, c_n) = self.lstm(x)

        # last hidden state from top LSTM layer
        out = h_n[-1]                     # batch x hidRNN
        out = self.dropout_layer(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout_layer(out)
        out = self.fc2(out)

        # residual connection from last observed timestep
        out = out + x[:, -1, :]

        if self.output is not None:
            out = self.output(out)

        return out