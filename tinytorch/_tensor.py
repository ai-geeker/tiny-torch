import numpy as np
from tinytorch.autograd import Function

class Engine:
    def __init__(self):
        pass

    def run_backward(self, grad_fn, grad_tensors):
        print("------", grad_fn)
        if grad_fn:
            if grad_tensors is None:
                grad_tensors = constant([1.])
            out = grad_fn.apply(grad_tensors)

            if isinstance(out, Tensor):
                out = [out]

            print(out)
            dirty_tensors = grad_fn.dirty_tensors
            if len(dirty_tensors) != len(out):
                print(" len(dirty_tensors) != len(out)")
                exit()

            for i in range(len(out)):
                dirty_tensors[i].backward(out[i])


engine = Engine()


class Tensor:
    # wrapper function for 2
    class Wrapper:
        def __init__(self, func):
            self.func = func

        def __call__(self, *args, **kwargs):
            output_requires_grad = False
            output_grad_fn = None

            dirty_tensors = []
            for arg in args:
                if arg.requires_grad:
                    output_requires_grad = True
                dirty_tensors.append(arg)

            output_grad_fn = self.func._backward_cls()
            output_grad_fn.mark_dirty(*dirty_tensors)
            output = self.func.forward(output_grad_fn, *args, **kwargs)
            if output_requires_grad:
                output.requires_grad = True
                output.grad_fn = output_grad_fn
            return output

    def __init__(self, data=None, requires_grad=False):
        if isinstance(data, Tensor):
            self.data = data.data
        else:
            self.data = data
        self._grad_ = requires_grad
        self.requires_grad = False
        self.grad_fn = None
        self.is_leaf = True

    @property
    def grad(self):
        return self._grad_

    def __str__(self):
        data_str = '<Tensor ' + str(self.data)
        if self.requires_grad:
            data_str += " , requires_grad = True"
            if self._grad_:
                data_str += " , grad = " + str(self._grad_)
        if self.grad_fn:
            data_str += " , " + str(self.grad_fn)
        data_str += '>'
        return data_str

    @staticmethod
    def convert(other):
        if isinstance(other, Tensor):
            return other
        else:
            return constant(other)

    def __add__(self, other):
        wrapper = Tensor.Wrapper(Add())
        return wrapper(self, Tensor.convert(other))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        wrapper = Tensor.Wrapper(Sub())
        return wrapper(self, Tensor.convert(other))

    def __rsub__(self, other):
        wrapper = Tensor.Wrapper(Sub())
        return wrapper(Tensor.convert(other), self)

    def __neg__(self):
        wrapper = Tensor.Wrapper(Neg())
        return wrapper(self)

    def __pow__(self, other):
        wrapper = Tensor.Wrapper(Pow())
        return wrapper(self, Tensor.convert(other))

    def __rpow__(self, other):
        wrapper = Tensor.Wrapper(Pow())
        return wrapper(Tensor.convert(other), self)

    def __mul__(self, other):
        wrapper = Tensor.Wrapper(Mul())
        return wrapper(self, Tensor.convert(other))

    def __rmul__(self, other):
        wrapper = Tensor.Wrapper(Mul())
        return wrapper(Tensor.convert(other), self)

    def __truediv__(self, other):
        wrapper = Tensor.Wrapper(Div())
        return wrapper(self, Tensor.convert(other))

    def __rtruediv__(self, other):
        wrapper = Tensor.Wrapper(Div())
        return wrapper(Tensor.convert(other), self)

    def __matmul__(self, other):
        wrapper = Tensor.Wrapper(Matmul())
        return wrapper(self, Tensor.convert(other))

    def sum(self):
        wrapper = Tensor.Wrapper(Sum())
        return wrapper(self)

    def mean(self):
        wrapper = Tensor.Wrapper(Mean())
        return wrapper(self)

    def max(self):
        wrapper = Tensor.Wrapper(Max())
        return wrapper(self)

    def min(self):
        wrapper = Tensor.Wrapper(Min())
        return wrapper(self)

    def backward(self, gradient=None):
        if not self.requires_grad:
            return
        if self.grad_fn:
            engine.run_backward(self.grad_fn, gradient)
        else:
            print("self._grad_", self._grad_, "gradient", gradient)
            if self._grad_ is None:
                self._grad_ = gradient
            else:
                self._grad_ = self._grad_ + gradient


def constant(v):
    output = Tensor()
    output.is_leaf = True
    output.data = np.array(v)
    return output


def ones(size, requires_grad=False):
    output = Tensor()
    output.is_leaf = True
    output.requires_grad = requires_grad
    output.data = np.ones(size)
    return output


def zeros(size, requires_grad=False):
    output = Tensor()
    output.requires_grad = requires_grad
    output.data = np.zeros(size)
    return output


def empty(size, requires_grad=False):
    output = Tensor()
    output.requires_grad = requires_grad
    output.data = np.empty(size)
    return output

def randn(size, requires_grad=False):
    output: Tensor = Tensor()
    output.requires_grad = requires_grad
    if type(size).__name__ == 'tuple':
        output.data = np.random.randn(*size)
    else:
        output.data = np.random.randn(size)
    return output


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        output = Tensor()
        output.data = a.data + b.data
        # ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


class Sum(Function):
    @staticmethod
    def forward(ctx, i):
        output = Tensor()
        output.data = i.data.sum()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output = Tensor()
        output.data = grad_output.data
        return output


class Neg(Function):
    @staticmethod
    def forward(ctx, i):
        output = Tensor()
        output.data = -i.data
        # ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output = Tensor()
        output.data = -grad_output.data
        return output


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        output = Tensor()
        output.data = a.data - b.data
        return output

    @staticmethod
    def backward(ctx, grad_output):
        neg_grad_output = Tensor()
        neg_grad_output.data = -grad_output.data
        return grad_output, neg_grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        output = Tensor()
        output.data = a.data * b.data
        ctx.save_for_backward(a, b)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_a_output = Tensor()
        grad_b_output = Tensor()
        a, b = ctx.saved_tensors
        grad_a_output.data = b.data * grad_output.data
        grad_b_output.data = a.data * grad_output.data
        return grad_a_output, grad_b_output


class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        output = Tensor()
        output.data = a.data / b.data
        ctx.save_for_backward(a, b)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_a_output = Tensor()
        grad_b_output = Tensor()
        a, b = ctx.saved_tensors
        grad_a_output.data = 1 / b.data * grad_output.data
        grad_b_output.data = - a.data / (b.data * b.data) * grad_output.data
        return grad_a_output, grad_b_output


class Matmul(Function):
    @staticmethod
    def forward(ctx, w, x):
        output = Tensor()
        output.data = w.data @ x.data
        ctx.save_for_backward(w, x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_w_output = Tensor()
        grad_x_output = Tensor()
        w, x = ctx.saved_tensors
        grad_w_output.data = x.data.T * grad_output.data
        grad_x_output.data = w.data * grad_output.data
        return grad_w_output, grad_x_output


class Pow(Function):
    @staticmethod
    def forward(ctx, a, b):
        result = Tensor()
        result.data = np.pow(a.data, b.data)
        ctx.save_for_backward(a, b, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        a, b, result = ctx.saved_tensors
        dy_da = Tensor()
        dy_db = Tensor()
        # z(1-z)
        # dy/da = b * a^b / a
        dy_da.data = (b.data * np.pow(a.data, b.data - 1)) * grad_output.data
        dy_db.data = result.data * np.log(np.a) * grad_output.data
        return dy_da, dy_db

class Mean(Function):
    @staticmethod
    def forward(ctx, i):
        output = Tensor()
        output.data = i.data.mean()
        # save the N
        ctx.save_for_backward(i.data.size)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output = Tensor()
        N, = ctx.saved_tensors
        output.data = grad_output.data / N
        return output


class Max(Function):
    @staticmethod
    def forward(ctx, i):
        output = Tensor()
        output.data = i.data.min()
        # save the N
        ctx.save_for_backward(i.data.size)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output = Tensor()
        N, = ctx.saved_tensors
        output.data = grad_output.data / N
        return output


class Min(Function):
    @staticmethod
    def forward(ctx, i):
        output = Tensor()
        output.data = i.data.min()
        # save the N
        ctx.save_for_backward(i.data.size)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output = Tensor()
        N, = ctx.saved_tensors
        output.data = grad_output.data / N
        return output

