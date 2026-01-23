from cv2.typing import MatLike
from dataclasses import dataclass


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


