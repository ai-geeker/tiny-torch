import numpy as np
from tinytorch.autograd import Function
from ._tensor import Tensor


class Exp(Function):
    @staticmethod
    def forward(ctx, i):
        result = Tensor()
        result.data = np.exp(i.data)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        ret = Tensor()
        ret.data = result.data * grad_output.data
        return ret


class Sin(Function):
    @staticmethod
    def forward(ctx, i):
        result = Tensor()
        result.data = np.sin(i.data)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        ret = Tensor()
        ret.data = np.cos(result.data) * grad_output.data
        return ret


def exp(i):
    x = Tensor.convert(i)
    return Tensor.Wrapper(Exp())(x)


def sin(i):
    x = Tensor.convert(i)
    return Tensor.Wrapper(Sin())(x)


def cos(i):
    x = Tensor.convert(i)
    return Tensor.Wrapper(Cos())(x)


class Cos(Function):
    @staticmethod
    def forward(ctx, i):
        result = Tensor()
        result.data = np.cos(i.data)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        ret = Tensor()
        ret.data = -(np.sin(result.data) * grad_output.data)
        return ret


def sigmoid(i):
    x = Tensor.convert(i)
    return Tensor.Wrapper(Sigmoid())(x)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, i):
        result = Tensor()
        z = i.data
        result.data = 1 / (1 + np.exp(-z))
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        ret = Tensor()
        # z(1-z)
        ret.data = result.data * (1 - result.data) * grad_output.data
        return ret


def _pow(a, b):
    a = Tensor.convert(a)
    b = Tensor.convert(b)
    return a ** b
