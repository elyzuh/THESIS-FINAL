import torch
import torch.nn as nn
from torch.nn import Parameter


class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()

        self.use_cuda = args.cuda
        self.m = data.m

        # ARIMA parameters
        self.p = args.window      # AR order
        self.d = getattr(args, "diff", 1)   # differencing order
        self.q = getattr(args, "ma", 1)     # MA order

        # AR coefficients (per variable)
        # shape: (p, m)
        self.ar_weight = Parameter(torch.Tensor(self.p, self.m))

        # MA coefficients
        # shape: (q, m)
        self.ma_weight = Parameter(torch.Tensor(self.q, self.m))

        # bias term
        self.bias = Parameter(torch.zeros(self.m))

        nn.init.xavier_normal_(self.ar_weight.unsqueeze(0))
        nn.init.xavier_normal_(self.ma_weight.unsqueeze(0))

        self.output = None
        if args.output_fun == 'sigmoid':
            self.output = torch.sigmoid
        elif args.output_fun == 'tanh':
            self.output = torch.tanh

    def difference(self, x):
        """
        Apply differencing d times along time dimension.
        x: [batch, window, m]
        """
        for _ in range(self.d):
            x = x[:, 1:, :] - x[:, :-1, :]
        return x

    def forward(self, x):
        """
        x: [batch, window, m]
        returns: [batch, m]
        """

        batch_size = x.size(0)

        # apply differencing
        if self.d > 0:
            x = self.difference(x)

        # AR component
        y = torch.zeros(batch_size, self.m, device=x.device)

        for l in range(min(self.p, x.size(1))):
            y += x[:, -(l + 1), :] * self.ar_weight[l]

        # MA component (approx using residuals as zero)
        # since residuals are unknown during forward
        for l in range(self.q):
            pass

        y = y + self.bias

        if self.output is not None:
            y = self.output(y)

        return y