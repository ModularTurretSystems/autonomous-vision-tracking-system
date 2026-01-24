from dataclasses import dataclass
from src.utils.image import combine_and_resize_frames, combine_frames, resize_frame

from src.camera.types import CameraFrame
from numpy import float32

from cv2.typing import MatLike, Size
from typing import List, Tuple
from numpy.typing import NDArray


@dataclass
class StereoFrame:
    camera_frame_l: CameraFrame
    camera_frame_r: CameraFrame


    def to_list(self) -> list[MatLike]:
        return [self.camera_frame_l.frame, self.camera_frame_r.frame]
    

    def copy(self):
        return StereoFrame(camera_frame_l=self.camera_frame_l.copy(), camera_frame_r=self.camera_frame_r.copy())    
    

    def flip(self, flip_code: int, in_place: bool = True) -> tuple[MatLike, MatLike]:
        return self.camera_frame_l.flip(flip_code=flip_code, in_place=in_place), self.camera_frame_r.flip(flip_code=flip_code, in_place=in_place)

    
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
    obj_points: List[NDArray[float32]]
    img_points_left: List[MatLike]
    img_points_right: List[MatLike]
    saved_images_count: int
    image_size: Tuple[int, int]
