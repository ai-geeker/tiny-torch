from ._tensor import Tensor

class Parameter(Tensor):
    pass

class Module:
    def __init__(self):
        pass

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        # 对Parameter类似的value，进行注册

    def __getattr__(self, name):
        return self.__dict__[name]

    def __delattr__(self, name):
        self.__dict__.pop(name)

