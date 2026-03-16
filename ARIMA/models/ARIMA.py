import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()

        self.use_cuda = args.cuda
        self.m = data.m
        self.w = args.window   # AR order (p)

        # AR weights
        self.ar_weight = Parameter(torch.Tensor(self.w, self.m))

        # MA weights (approximate residual influence)
        self.ma_weight = Parameter(torch.Tensor(self.w, self.m))

        # bias
        self.bias = Parameter(torch.zeros(self.m))

        nn.init.xavier_normal_(self.ar_weight)
        nn.init.xavier_normal_(self.ma_weight)

        self.output = None
        if args.output_fun == 'sigmoid':
            self.output = F.sigmoid
        if args.output_fun == 'tanh':
            self.output = F.tanh

    def forward(self, x):
        # x: batch x window x m
        batch_size = x.size(0)

        # AR component
        ar_part = torch.sum(x * self.ar_weight, dim=1)

        # MA component (approximation using previous errors)
        residual = x[:, -1, :] - ar_part
        ma_part = residual * self.ma_weight[0]

        y = ar_part + ma_part + self.bias

        if self.output is not None:
            y = self.output(y)

        return y