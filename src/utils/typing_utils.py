"""
src/utils/typing_utils.py

Author
------
KrutayaBabka : initial implementation (2026-02-12)

Description
-----------
Utility functions for type conversion between PyTorch tensors and NumPy arrays.
"""

import numpy as np
from torch import Tensor

from typing import Union


def to_ndarray(arr: Union[Tensor, np.ndarray]) -> np.ndarray:
    """
    Convert input data to a NumPy ndarray.

    This function accepts either a PyTorch Tensor or a NumPy ndarray and
    returns a NumPy array. If a Tensor is provided, it is first moved to
    CPU and converted to a NumPy array. If the input is already a NumPy
    array, it is returned as-is.

    Parameters
    ----------
    arr : Union[Tensor, np.ndarray]
        Input data to convert.

    Returns
    -------
    np.ndarray
        The converted NumPy array.

    Notes
    -----
    - This function does not modify the original input.
    - Useful when a unified interface is required for downstream
      NumPy-based processing pipelines.
    """

    if isinstance(arr, Tensor):
        return arr.cpu().numpy()

    if isinstance(arr, np.ndarray): # type: ignore
        return arr

    raise TypeError(f"Expected Tensor or np.ndarray, got {type(arr).__name__}")
