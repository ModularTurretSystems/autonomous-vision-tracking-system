import cv2
from pathlib import Path

from src.camera.camera import Camera

from src.camera.types import CameraFrame

from cv2.typing import MatLike
from typing import overload


class MonoCapture:
    @overload
    def __init__(self, camera: Camera, *, save_dir: Path | str | None = None) -> None: ...


    @overload
    def __init__(self, camera: int, *, save_dir: Path | str | None = None) -> None: ...


    def __init__(self, camera: Camera | int, *, save_dir: Path | str | None = None) -> None:
        if isinstance(camera, Camera): self.camera = camera
        if isinstance(camera, int): self.camera = Camera(camera_id=camera)

        if save_dir is not None: self.set_save_dir(save_dir=save_dir)
        else: self.save_dir = None

        self._saved_frame_count = 0


    def set_save_dir(self, save_dir: Path | str) -> None:
        p = Path(save_dir)

        if not p.exists():
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise OSError(f"Failed to create directory {p}: {e}") from e
        elif not p.is_dir():
            raise NotADirectoryError(f"The path exists but is not a directory: {p}")

        self.save_dir = p


    def capture_frame(self) -> CameraFrame:
        return self.camera.capture_frame()
    

    def save_frame(self, frame: MatLike, base_name: str, ext: str = "png") -> bool:
        if self.save_dir is None:
            raise ValueError("Save directory is not set. Use `set_save_dir` before saving frames.")
        
        filename = f"{self.save_dir}/{base_name}_{self._saved_frame_count + 1}.{ext}"

        cv2.imwrite(filename=filename, img=frame)
        self._saved_frame_count += 1

        return True
    

    def get_saved_frame_count(self) -> int:
        return self._saved_frame_count
