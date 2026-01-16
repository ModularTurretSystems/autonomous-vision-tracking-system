import cv2
from dataclasses import dataclass
from .constants import(
    CameraProperty,
    ApiPreference
)

from cv2 import(
    VideoCapture
)
from cv2.typing import(
    MatLike
)
from typing import(
    Sequence
)


@dataclass
class Property:
    name: str
    value: float


class Camera:
    def __init__(
        self, 
        camera_id: int = 0, 
        apiPreference: int | None = None, 
        params: Sequence[int] | None = None
    ) -> None:
        self.camera_id = camera_id
        self.apiPreference = apiPreference
        self.params = params

        if apiPreference:
            if params:
                self.cap = cv2.VideoCapture(index=camera_id, apiPreference=apiPreference, params=params)
            else:
                self.cap = cv2.VideoCapture(index=camera_id, apiPreference=apiPreference)
        else:
            self.cap = cv2.VideoCapture(index=camera_id)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera with id {camera_id}")
                

    def get_camera_properties(self) -> dict[int, Property]:
        properties: dict[int, Property] = {
            property.value: Property(name=property.name, value=self.cap.get(propId=property.value))
            for property in CameraProperty
        }

        return properties
     

    def print_camera_properties(self) -> None:
        for property in self.get_camera_properties().values(): print(property.name, property.value)


    @classmethod
    def get_supported_api_preferences(cls, camera_id: int = 0) -> dict[int, str]:
        log_lvl_of_cv2: int = cv2.setLogLevel(level=0) # To config!!!

        working_apis: dict[int, str] = {}
        for api in ApiPreference:
            cap = cv2.VideoCapture(index=camera_id, apiPreference=api.value)
            if cap.isOpened():
                working_apis[api.value] = api.name
                cap.release()

        cv2.setLogLevel(level=log_lvl_of_cv2)

        return working_apis
    

    @classmethod
    def print_supported_api_preferences(cls, camera_id: int = 0) -> None:
        working_apis: dict[int, str] = cls.get_supported_api_preferences(camera_id=camera_id)
        for key, value in working_apis.items(): print(key, value)


    def capture_frame(self) -> tuple[bool, MatLike]:
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Failed to capture frame from camera {self.camera_id}")
        
        return ret, frame


    def set(self, propId: int, value: float) -> bool:        
        ret = self.cap.set(propId=propId, value=value)
        return ret and self.cap.get(propId=propId) == value
    

    def get(self, propId: int) -> float:
        return self.cap.get(propId=propId)


    def release(self) -> None:
        self.cap.release()

    
    def __del__(self):
        try:
            self.release()
        except Exception as e:
            print(f"Exception in __del__: {e}")        
