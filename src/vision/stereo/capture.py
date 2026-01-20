from pathlib import Path
import cv2
from cv2.typing import MatLike
from .types import StereoFrame
from .system import StereoSystem
from src.utils.image import combine_frames, resize_frame


class StereoCapture:
    def __init__(
            self, 
            stereo_system: StereoSystem,
            save_dir: Path | str,
    ) -> None:
        self.stereo_system = stereo_system

        self.save_dir: Path = Path(save_dir) if isinstance(save_dir, str) else save_dir

        self.save_path_left: Path = self.save_dir / "left_cam"
        self.save_path_right: Path = self.save_dir / "right_cam"

        self.save_path_left.mkdir(parents=True, exist_ok=True)
        self.save_path_right.mkdir(parents=True, exist_ok=True)

        self.number_of_frames: int = 0


    def get_number_of_frames(self) -> int:
        return self.number_of_frames
    

    def capture_frame(self) -> StereoFrame:
        return self.stereo_system.capture_frame()
    

    def get_combined_frame(self, horizontal: bool = True) -> MatLike:
        stereo_frame = self.capture_frame()
        return combine_frames(frames=stereo_frame.to_list(), horizontal=horizontal)


    def get_combined_and_resized_frame(self, frame_size: tuple[int, int], horizontal: bool = True) -> MatLike:
        combined_frame = self.get_combined_frame(horizontal=horizontal)
        return resize_frame(img=combined_frame, frame_size=frame_size)
    

    def save_frame(
            self, 
            frame: StereoFrame
    ) -> None:        
        left_name: str = f"{self.save_path_left}/{self.number_of_frames}.jpg"
        right_name: str = f"{self.save_path_right}/{self.number_of_frames}.jpg"

        cv2.imwrite(filename=left_name, img=frame.frame_l)
        cv2.imwrite(filename=right_name, img=frame.frame_r)

        self.number_of_frames += 1


    def __enter__(self):
        return self
    

    def __exit__(self, exc_type, exc_val, exc_tb) -> None: #type: ignore
        pass
