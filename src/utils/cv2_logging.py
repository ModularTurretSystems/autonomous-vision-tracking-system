import cv2
from functools import wraps

from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable[..., object])


def suppress_cv2_logs(level: int = 0):
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs): # type: ignore
            old_level = cv2.setLogLevel(level)
            try:
                return func(*args, **kwargs)
            finally:
                cv2.setLogLevel(old_level)
        return wrapper  # type: ignore
    return decorator
