from ..autograd import Function
from .._tensor import Tensor
import numpy as np


def relu(x):
    return Tensor.Wrapper(Relu())(x)


class Relu(Function):
    @staticmethod
    def forward(ctx, i):
        result = Tensor()
        k = (i.data > 0)
        result.data = i.data * k
        ctx.save_for_backward(k)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        k, = ctx.saved_tensors
        ret = Tensor()
        ret.data = k * grad_output.data
        return ret






