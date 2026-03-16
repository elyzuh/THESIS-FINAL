import torch
import torch.nn as nn
from torch.nn import Parameter


class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()

        self.use_cuda = args.cuda
        self.m = data.m
        self.w = args.window
        self.hidR = args.hidRNN
        self.n_layers = getattr(args, 'LSTM_layers', 1)
        self.dropout = getattr(args, 'dropout', 0.2)

        self.lstm = nn.LSTM(
            input_size=self.m,
            hidden_size=self.hidR,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=self.dropout if self.n_layers > 1 else 0
        )

        self.linear = nn.Linear(self.hidR, self.m)

        self.output = None
        if args.output_fun == 'sigmoid':
            self.output = torch.sigmoid
        elif args.output_fun == 'tanh':
            self.output = torch.tanh

    def forward(self, x):
        # x: batch x window x m

        lstm_out, (h_n, c_n) = self.lstm(x)

        # use last hidden state from final LSTM layer
        x = h_n[-1]

        x = self.linear(x)

        if self.output is not None:
            x = self.output(x)

        return x