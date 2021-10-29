from itertools import chain

import torch
from abc import ABC, abstractmethod

from functools import update_wrapper, partial


class Decorator(ABC):
    def __init__(self, f):
        self.func = f
        update_wrapper(self, f, updated=[])  # updated=[] so that 'self' attributes are not overwritten

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def __get__(self, instance, owner):
        new_f = partial(self.__call__, instance)
        update_wrapper(new_f, self.func)
        return new_f


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    else:
        return x


# noinspection PyPep8Naming
class input_to_numpy(Decorator):
    def __call__(self, *args, **kwargs):
        new_args = [to_numpy(arg) for arg in args]
        new_kwargs = {key: to_numpy(value) for key, value in kwargs.items()}
        return self.func(*new_args, **new_kwargs)


# noinspection PyPep8Naming
class output_to_numpy(Decorator):
    def __call__(self, *args, **kwargs):
        outputs = self.func(*args, **kwargs)
        if isinstance(outputs, torch.Tensor):
            return to_numpy(outputs)
        if isinstance(outputs, tuple):
            new_outputs = tuple([to_numpy(item) for item in outputs])
            return new_outputs
        return outputs


# noinspection PyPep8Naming
class none_if_missing_arg(Decorator):
    def __call__(self, *args, **kwargs):
        for arg in chain(args, kwargs.values()):
            if arg is None:
                return None

        return self.func(*args, **kwargs)
