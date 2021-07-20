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

def mse_loss(input, target, reduction="mean"):
    loss = MSELoss()
    if reduction is None or len(reduction) == 0:
        reduction = "mean"

    loss.reduction = reduction
    return Tensor.Wrapper(loss)(input, target)

class MSELoss(Function):
    @staticmethod
    def forward(ctx, input, target):
        result = Tensor()
        s = np.square(input.data - target.data)
        N = input.data.size
        if ctx.reduction == "mean":
            result.data = s.mean()
        else:
            result.data = s.sum()

        ctx.save_for_backward(input.data - target.data)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        in_sub_target, = ctx.saved_tensors
        input_grad = Tensor()
        out_grad = Tensor()
        input_grad.data = 2 * in_sub_target * grad_output.data
        out_grad.data = -2 * in_sub_target * grad_output.data

        if ctx.reduction == "mean":
            input_grad.data = input_grad.data / input_grad.data.size
            out_grad.data = out_grad.data / input_grad.data.size
        return input_grad, out_grad.data



