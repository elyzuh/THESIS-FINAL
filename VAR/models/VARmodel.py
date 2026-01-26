import torch
import torch.nn as nn
from torch.nn import Parameter


class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()

        self.use_cuda = args.cuda
        self.m = data.m
        self.p = args.window   # VAR order

        # VAR coefficients: one m×m matrix per lag
        # Shape: (p, m, m)
        self.weight = Parameter(torch.Tensor(self.p, self.m, self.m))

        # Bias per variable
        self.bias = Parameter(torch.zeros(self.m))

        nn.init.xavier_normal_(self.weight)

        self.output = None
        if args.output_fun == 'sigmoid':
            self.output = torch.sigmoid
        elif args.output_fun == 'tanh':
            self.output = torch.tanh

    def forward(self, x):
        """
        x: [batch, window (p), m]
        returns: [batch, m]
        """
        batch_size = x.size(0)

        # VAR computation
        y = torch.zeros(batch_size, self.m, device=x.device)

        for l in range(self.p):
            # x[:, l, :] @ A_l
            y += torch.matmul(x[:, l, :], self.weight[l])

        y = y + self.bias

        if self.output is not None:
            y = self.output(y)

        return y
