import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()

        self.use_cuda = args.cuda
        self.m = data.m
        self.w = args.window

        self.lstm1 = nn.LSTM(
            input_size=self.m,
            hidden_size=50,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(0.2)

        self.lstm2 = nn.LSTM(
            input_size=50,
            hidden_size=60,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(0.3)

        self.lstm3 = nn.LSTM(
            input_size=60,
            hidden_size=80,
            batch_first=True
        )
        self.dropout3 = nn.Dropout(0.4)

        self.lstm4 = nn.LSTM(
            input_size=80,
            hidden_size=120,
            batch_first=True
        )
        self.dropout4 = nn.Dropout(0.5)

        self.linear = nn.Linear(120, self.m)

        self.output = None
        if args.output_fun == 'sigmoid':
            self.output = torch.sigmoid
        elif args.output_fun == 'tanh':
            self.output = torch.tanh

    def forward(self, x):
        # x: batch x window x m

        x, _ = self.lstm1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x, _ = self.lstm3(x)
        x = torch.relu(x)
        x = self.dropout3(x)

        x, _ = self.lstm4(x)
        x = torch.relu(x)
        x = self.dropout4(x)

        x = x[:, -1, :]
        x = self.linear(x)

        if self.output is not None:
            x = self.output(x)

        return x