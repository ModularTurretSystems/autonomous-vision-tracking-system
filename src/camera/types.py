import cv2
from dataclasses import dataclass

from cv2.typing import MatLike


@dataclass
class Property:
    name: str
    value: float
    

@dataclass
class CameraFrame:
    success: bool
    frame: MatLike


    def copy(self):
        return CameraFrame(success=self.success, frame=self.frame.copy())
    

    def flip(self, flip_code: int, in_place: bool = True) -> MatLike:
        dst = self.frame if in_place else None
        return cv2.flip(src=self.frame, flipCode=flip_code, dst=dst)
