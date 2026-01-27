import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()

        self.use_cuda = args.cuda
        self.m = data.m
        self.w = args.window
        self.hidRNN = args.hidRNN

        # GAR uses a neural network to learn nonlinear dependencies
        # Input: window x m features
        # Hidden layer for feature extraction
        self.fc1 = nn.Linear(self.w * self.m, self.hidRNN)
        self.dropout = nn.Dropout(args.dropout)

        # Output layer
        self.fc2 = nn.Linear(self.hidRNN, self.m)

        self.output = None
        if args.output_fun == 'sigmoid':
            self.output = F.sigmoid
        elif args.output_fun == 'tanh':
            self.output = F.tanh

    def forward(self, x):
        # x: batch x window (self.w) x #signal (m)
        batch_size = x.size(0)

        # Flatten the input: batch x (window * m)
        x = x.view(batch_size, -1)

        # Hidden layer with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Output layer
        x = self.fc2(x)

        if self.output is not None:
            x = self.output(x)

        return x
