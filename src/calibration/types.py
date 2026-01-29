from dataclasses import dataclass
from cv2.typing import MatLike
from typing import Tuple

@dataclass
class CalibratedData:
    rms: float
    matx_l: MatLike
    dist_l: MatLike
    matx_r: MatLike
    dist_r: MatLike
    R: MatLike
    T: MatLike
    E: MatLike
    F: MatLike

@dataclass
class FinalCalibratedData:
    CalibratedData: CalibratedData
    map_l: Tuple[MatLike, MatLike]
    map_r: Tuple[MatLike, MatLike]
    Q: MatLike
    image_size: Tuple[int, int]
    