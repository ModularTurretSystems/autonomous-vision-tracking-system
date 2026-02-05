from src.camera.camera import Camera
from .types import StereoFrame


class StereoSystem:
    def __init__(
        self,
        left: Camera | int,
        right: Camera | int,
    ) -> None:
        if isinstance(left, Camera):
            self.left_cam = left
            self._should_release_left_cam = False
        else:
            self.left_cam = Camera(camera_id=left)
            self._should_release_left_cam = True
            
        if isinstance(right, Camera):
            self.right_cam = right
            self._should_release_right_cam = False
        else:
            self.right_cam = Camera(camera_id=right)
            self._should_release_right_cam = True

        if self.left_cam.camera_id == self.right_cam.camera_id:
            raise ValueError(
                "StereoSystem requires two different physical cameras "
                f"(got camera_id={self.left_cam.camera_id} for both)"
            )
        

    def capture_frame(self) -> StereoFrame:
        res_l = self.left_cam.capture_frame()
        res_r = self.right_cam.capture_frame()
        
        return StereoFrame(camera_frame_l=res_l, camera_frame_r=res_r)
    

    def release(self) -> None:
        self.left_cam.release()
        self.right_cam.release()


    def __enter__(self):
        return self
    

    def __exit__(self, exc_type, exc_val, exc_tb) -> None: #type: ignore
        if self._should_release_left_cam:
            self.left_cam.release()

        if self._should_release_right_cam:
            self.right_cam.release()


    def __del__(self) -> None:
        try:
            if self._should_release_left_cam:
                self.left_cam.release()

            if self._should_release_right_cam:
                self.right_cam.release()

        except Exception as e:
            print(f"Exception in __del__: {e}")  
