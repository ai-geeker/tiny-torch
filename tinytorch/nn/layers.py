import numpy as np
import tinytorch as torch
from tinytorch.nn import Module


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = torch.nn.Parameter(torch.randn((out_features, in_features)))
        print(self.W.data)
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        if self.bias:
            return self.W @ input + self.bias
        else:
            return input @ self.W

