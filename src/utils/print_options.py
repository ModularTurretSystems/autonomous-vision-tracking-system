import numpy as np
from functools import wraps

from typing import Callable, TypeVar


F = TypeVar("F", bound=Callable[..., object])


def np_printoptions(precision: int = 4, suppress: bool = True):
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs): # type: ignore
            old_opts = np.get_printoptions()
            np.set_printoptions(precision=precision, suppress=suppress)
            try:
                return func(*args, **kwargs)
            finally:
                np.set_printoptions(**old_opts)
        return wrapper  # type: ignore
    return decorator
