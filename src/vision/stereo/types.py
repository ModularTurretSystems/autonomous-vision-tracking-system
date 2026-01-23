from dataclasses import dataclass
from cv2.typing import MatLike, Size
from src.utils.image import combine_and_resize_frames, combine_frames, resize_frame
import cv2
from typing import List, Tuple
import numpy as np

@dataclass
class StereoFrame:
    frame_l: MatLike
    frame_r: MatLike


    def to_list(self) -> list[MatLike]:
        return [self.frame_l, self.frame_r]
    

    def copy(self):
        return StereoFrame(frame_l=self.frame_l.copy(), frame_r=self.frame_r.copy())
    

    def flip(self, flipCode: int) -> None:
        cv2.flip(src=self.frame_l, flipCode=flipCode, dst=self.frame_l)
        cv2.flip(src=self.frame_r, flipCode=flipCode, dst=self.frame_r)

    
    def combine_frames(self, horizontal: bool = True) -> MatLike:
        return combine_frames(frames=self.to_list(), horizontal=horizontal)
    

    @classmethod
    def resize(cls, img: MatLike, new_size: Size) -> MatLike:
        return resize_frame(img=img, frame_size=new_size)
    

    def combined_and_resize_frames(self, new_size: tuple[int, int], horizontal: bool = True) -> MatLike:
        return combine_and_resize_frames(frame_size=new_size, frames=self.to_list(), horizontal=horizontal)

@dataclass
class RawData:
    collected: int
    enough: bool
    obj_points: List[np.ndarray]
    img_points_left: List[np.ndarray]
    img_points_right: List[np.ndarray]
    saved_images_count: int
    image_size: Tuple[int, int]

