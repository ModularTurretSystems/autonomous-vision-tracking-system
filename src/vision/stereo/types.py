from dataclasses import dataclass
from cv2.typing import MatLike


@dataclass
class StereoFrame:
    frame_l: MatLike
    frame_r: MatLike