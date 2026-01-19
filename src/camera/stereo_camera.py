import cv2, os
from src.camera.camera import Camera
from cv2.typing import MatLike
from src.camera.types import StereoFrames
class StereoCapture:
    
    def __init__(
            self, 
            left_id: int, 
            right_id: int, 
            width: int = 640, 
            height: int = 640
    ) -> None:
        #Инициализация камер
        self.cam_left = Camera(camera_id=left_id)
        self.cam_right = Camera(camera_id=right_id)

        #Подгоняем камеры под одно разрешение
        for cam in [self.cam_left, self.cam_right]:
            cam[cv2.CAP_PROP_FRAME_WIDTH] = width
            cam[cv2.CAP_PROP_FRAME_HEIGHT] = height

        #Создание папки, если нет
        self.save_path_left = "data/calibration_images/left_cam"
        self.save_path_right = "data/calibration_images/right_cam"
        os.makedirs(self.save_path_left, exist_ok=True)
        os.makedirs(self.save_path_right, exist_ok=True)

    def get_frames(self) -> StereoFrames:
        
        #Получение кадров с камер
        _, frame_left = self.cam_left.capture_frame()
        _, frame_right = self.cam_right.capture_frame()

        return StereoFrames(left=frame_left, right=frame_right)
    
    def save_pairs(
            self, 
            frame_left: MatLike, 
            frame_right: MatLike, 
            count: int
    ) -> None:
        
        #Пути для сохранения изображения
        left_name = f"{self.save_path_left}/{count}.jpg"
        right_name = f"{self.save_path_right}/{count}.jpg"

        #Запись
        cv2.imwrite(filename=left_name, img=frame_left)
        cv2.imwrite(filename=right_name, img=frame_right)
    
    def release(self) -> None:
        #Закрытие камер
        self.cam_left.release()
        self.cam_right.release()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()