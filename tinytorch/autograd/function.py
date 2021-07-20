import numpy as np


def with_metaclass(meta: type, *bases) -> type:
    """Create a base class with a metaclass."""

    # This requires a bit of explanation: the basic idea is to make a dummy
    # metaclass for one level of class instantiation that replaces itself with
    # the actual metaclass.
    class metaclass(meta):  # type: ignore[misc, valid-type]

        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)

        @classmethod
        def __prepare__(cls, name, this_bases):
            return meta.__prepare__(name, bases)

    return type.__new__(metaclass, 'temporary_class', (), {})


class _ContextMethodMixin(object):

    def save_for_backward(self, *tensors):
        r"""Saves given tensors for a future call to :func:`~Function.backward`.

        **This should be called at most once, and only from inside the**
        :func:`forward` **method.**

        Later, saved tensors can be accessed through the :attr:`saved_tensors`
        attribute. Before returning them to the user, a check is made to ensure
        they weren't used in any in-place operation that modified their content.

        Arguments can also be ``None``.
        """
        self.to_save = tensors

    def mark_dirty(self, *args):
        r"""Marks given tensors as modified in an in-place operation.

        **This should be called at most once, only from inside the**
        :func:`forward` **method, and all arguments should be inputs.**

        Every tensor that's been modified in-place in a call to :func:`forward`
        should be given to this function, to ensure correctness of our checks.
        It doesn't matter whether the function is called before or after
        modification.
        """
        self.dirty_tensors = args

class FunctionBase:
    def __init__(self):
        self.requires_grad = False

    @property
    def saved_tensors(self):
        return self.to_save

class BackwardCFunction(FunctionBase, _ContextMethodMixin):
    def apply(self, *args):
        # _forward_cls is defined by derived class
        return self._forward_cls.backward(self, *args)


class FunctionMeta(type):
    """Function metaclass.

    This metaclass sets up the following properties:
        _backward_cls: The Function class corresponding to the differentiated
            version of this function (which is generated on the fly by this
            metaclass).
    """

    def __init__(cls, name, bases, attrs):
        backward_fn = type(name + 'Backward', (BackwardCFunction,), {'_forward_cls': cls})
        cls._backward_cls = backward_fn

        return super(FunctionMeta, cls).__init__(name, bases, attrs)


class Function(with_metaclass(FunctionMeta, FunctionBase, _ContextMethodMixin)):
    def __init__(self):
        pass

    @staticmethod
    def forward(ctx, *args, **kwargs):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        pass
