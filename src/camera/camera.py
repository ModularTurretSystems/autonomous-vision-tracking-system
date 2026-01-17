import cv2
from .types import Property
from src.utils.cv2_logging import suppress_cv2_logs
from .constants import(
    CameraProperty,
    ApiPreference
)

from cv2.typing import(
    MatLike
)
from typing import(
    Sequence
)


class Camera:
    def __init__(
        self, 
        camera_id: int = 0, 
        apiPreference: int | None = None, 
        params: Sequence[int] | None = None,
        preserve_defaults: bool = False
    ) -> None:
        self.camera_id = camera_id
        self.apiPreference = apiPreference
        self.params = params
        self.preserve_default = preserve_defaults
        self._default_properies = {}

        if self.apiPreference:
            if self.params:
                self.cap = cv2.VideoCapture(index=self.camera_id, apiPreference=self.apiPreference, params=self.params)
            else:
                self.cap = cv2.VideoCapture(index=self.camera_id, apiPreference=self.apiPreference)
        else:
            self.cap = cv2.VideoCapture(index=self.camera_id)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera with id {camera_id}")
        
        if self.preserve_default:   
            self._default_properies = self.get_camera_properties()


    def get_camera_properties(self) -> dict[int, Property]:
        properties: dict[int, Property] = {
            property.value: Property(name=property.name, value=self.cap.get(propId=property.value))
            for property in CameraProperty
        }

        return properties
    

    @suppress_cv2_logs()
    def set_camera_properties(self, properties: dict[int, Property]) -> None:
        for key, property in properties.items(): 
            self.cap.set(propId=key, value=property.value)
        

    def print_camera_properties(self) -> None:
        for property in self.get_camera_properties().values(): print(property.name, property.value)


    @classmethod
    @suppress_cv2_logs()
    def get_supported_api_preferences(cls, camera_id: int = 0) -> dict[int, str]:
        working_apis: dict[int, str] = {}
        for api in ApiPreference:
            cap = cv2.VideoCapture(index=camera_id, apiPreference=api.value)
            if cap.isOpened():
                working_apis[api.value] = api.name
                cap.release()

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


    def _filter_settable_properties(self, properties: dict[int, Property]) -> dict[int, Property]:
        props = properties.copy()
        props.pop(cv2.CAP_PROP_BACKEND)
        return props
    

    def release(self) -> None:
        if self.preserve_default: self.set_camera_properties(properties=self._filter_settable_properties(properties=self._default_properies))

        self.cap.release()

    
    def __enter__(self):
        return self
    

    def __getitem__(self, propId: int) -> float:
        return self.get(propId=propId)
    

    def __setitem__(self, propId: int, value: float) -> None:
        self.set(propId=propId, value=value)


    def __exit__(self, exc_type, exc_value, traceback) -> None: # type: ignore
        self.release()  
    
    
    def __del__(self) -> None:
        try:
            self.release()
        except Exception as e:
            print(f"Exception in __del__: {e}")      
