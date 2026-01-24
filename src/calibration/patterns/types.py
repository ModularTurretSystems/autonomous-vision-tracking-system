from dataclasses import dataclass

from cv2.typing import MatLike
from typing import Sequence


@dataclass
class DetectionUnit:
    found: bool
    corners: MatLike


@dataclass
class DetectionData(DetectionUnit):
    ids: MatLike


@dataclass
class CharucoData(DetectionData):
    pass


@dataclass
class MarkerData:
    found: bool
    corners: Sequence[MatLike]
    ids: MatLike


@dataclass
class CharucoDetectionResult:
    charuco: CharucoData
    markers: MarkerData


@dataclass
class ChessBoardDetectionResult(DetectionUnit):
    pass
