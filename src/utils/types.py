from dataclasses import dataclass
from pickle import FRAME
from typing import Tuple

@dataclass
class AOVParameters:
    horizontal: float
    vertical: float
    diagonal: float

@dataclass(frozen=True)
class VisionConfig:
    CAM_L_ID: int = 0
    CAM_R_ID: int = 1

    FRAME_W: int = 1920
    FRAME_H: int = 1080

    DISPLAY_W: int = 1920
    DISPLAY_H: int = 1080

    @property
    def FRAME_SIZE(self) -> Tuple[int, int]:
        return (self.FRAME_W, self.FRAME_H)
    
    @property
    def DISPLAY_SIZE(self) -> Tuple[int, int]:
        return (self.DISPLAY_W, self.DISPLAY_H)

CFG = VisionConfig(FRAME_H=1080, FRAME_W=1920)