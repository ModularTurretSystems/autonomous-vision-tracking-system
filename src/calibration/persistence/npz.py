import numpy as np
from pathlib import Path

from dataclasses import fields, is_dataclass
from src.utils.path import ensure_file_path

from typing import TypeVar, Generic, Type


T = TypeVar("T")


class NpzCalibrationStorage(Generic[T]):
    def __init__(self, cls: Type[T]) -> None:
        if not is_dataclass(cls):
            raise ValueError("cls must be a dataclass")
        
        self._cls = cls


    def save(self, result: T, filename: Path | str) -> None:
        file_path = ensure_file_path(file_path=filename)

        data = {f.name: getattr(result, f.name) for f in fields(self._cls)}
        np.savez(file=file_path, **data)


    def load(self, filename: Path | str) -> T:
        file_path = ensure_file_path(file_path=filename)

        npz_data = np.load(file=file_path, allow_pickle=True)
        kwargs = {f.name: npz_data[f.name] for f in fields(self._cls)}

        return self._cls(**kwargs)
