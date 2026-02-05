import cv2
from pathlib import Path

from .system import StereoSystem
from src.utils.path import ensure_directory

from .constants import LEFT_CAMERA_DIR_NAME, RIGHT_CAMERA_DIR_NAME

from cv2.typing import MatLike
from .types import StereoFrame


class StereoCapture:
    def __init__(
            self, 
            stereo_system: StereoSystem,
            save_dir: Path | str,
    ) -> None:
        self.stereo_system = stereo_system

        self.save_dir = ensure_directory(path=save_dir)

        self.save_path_left: Path = ensure_directory(self.save_dir / LEFT_CAMERA_DIR_NAME)
        self.save_path_right: Path = ensure_directory(self.save_dir / RIGHT_CAMERA_DIR_NAME)

        self._saved_frame_count: int = 0


    def get_saved_frame_count(self) -> int:
        "Number of successfully saved pairs of frames"
        return self._saved_frame_count
    
    
    def capture_frame(self) -> StereoFrame:
        "Get a original stereo pair"
        return self.stereo_system.capture_frame()
    

    def get_combined_frame(self, horizontal: bool = True) -> MatLike:
        "Return the stacked frame"
        return self.capture_frame().combine_frames(horizontal=horizontal)


    def get_combined_and_resized_frame(self, frame_size: tuple[int, int], horizontal: bool = True) -> MatLike:
        "Return the stacked and resized frame"
        return self.capture_frame().combined_and_resize_frames(new_size=frame_size, horizontal=horizontal)
    

    def save_frame(
            self, 
            frame: StereoFrame,
            *,
            base_name: str | None = None,
            ext: str = "png"
    ) -> bool:
        """
        Save stereo pair with current number.
        Increments the counter only after successeful saving.
        """        
        base_name = '' if base_name is None else base_name + '_'
        
        left_filename: str = f"{self.save_path_left}/{base_name}{self.get_saved_frame_count() + 1}.{ext}"
        right_filename: str = f"{self.save_path_right}/{base_name}{self.get_saved_frame_count() + 1}.{ext}"

        cv2.imwrite(filename=left_filename, img=frame.camera_frame_l.frame)
        cv2.imwrite(filename=right_filename, img=frame.camera_frame_r.frame)
            
        self._saved_frame_count += 1

        return True


    def __enter__(self):
        return self
    

    def __exit__(self, exc_type, exc_val, exc_tb) -> None: #type: ignore
        pass
