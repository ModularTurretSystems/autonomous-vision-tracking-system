from typing import overload

@overload
def f(arg: int) -> int: ...

@overload
def f(arg: float) -> int: ...

def f(arg: int | float) -> int:
    return int(arg)
