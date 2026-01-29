from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar


T = TypeVar("T")


class CalibrationStorage(Generic[T], ABC):

    @abstractmethod
    def save(self, result: T, filename: Path) -> None:
        ...


    @abstractmethod
    def load(self, filename: Path) -> T:
        ...
