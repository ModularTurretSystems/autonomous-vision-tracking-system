from dataclasses import dataclass
from cv2.typing import MatLike

@dataclass
class Property:
    name: str
    value: float
    
@dataclass
class StereoFrames:
    left: MatLike
    right: MatLike