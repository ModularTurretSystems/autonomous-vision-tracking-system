from dataclasses import dataclass
from cv2.typing import MatLike


@dataclass
class StereoFrame:
    frame_l: MatLike
    frame_r: MatLike

    def to_list(self) -> list[MatLike]:
        return [self.frame_l, self.frame_r]
    