from dataclasses import dataclass
from typing import Tuple, Optional, Union
from torch.types import Tensor
from cv2.typing import MatLike

import numpy as np

@dataclass
class BBoxDet:
    xyxy: np.ndarray #x1 y1 x2 y2
    conf: float
    cls: int

    @property
    def center_x(self) -> float:
        x1, _, x2, _ = self.xyxy
        return float((x1 + x2) * 0.5)

    @property
    def center_y(self) -> float:
        _, y1, _, y2 = self.xyxy
        return float((y1 + y2) * 0.5)
    
    @property
    def w(self) -> float:
        x1, _, x2, _ = self.xyxy
        return float(x2 - x1)
    
    @property
    def h(self) -> float:
        _, y1, _, y2 = self.xyxy
        return float(y2 - y1)
    
    @property
    def area(self) -> float:
        return max(self.w, 0.0) * max(self.h, 0.0)

@dataclass
class Aim:
    x: float
    y: float

    @property
    def get_xy(self) -> Tuple[float, float]:
        return (self.x, self.y)

@dataclass(frozen=True)
class AimError:
    ex_px: int
    ey_px: int
    ex_norm: float
    ey_norm: float
    ex_ang: Optional[float] = None
    ey_ang: Optional[float] = None

@dataclass
class VisionData:
    map_l_x: MatLike
    map_l_y: MatLike
    map_r_x: MatLike
    map_r_y: MatLike
    Q: MatLike
    P1: MatLike
    
def to_numpy(x: Union[Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.cpu().numpy()
    return x