import cv2
from enum import IntEnum

from typing import ClassVar


class ChessboardFlags(IntEnum):
    DEFAULT = 0
    ADAPTIVE_THRESH = cv2.CALIB_CB_ADAPTIVE_THRESH
    NORMALIZE_IMAGE = cv2.CALIB_CB_NORMALIZE_IMAGE
    FAST_CHECK = cv2.CALIB_CB_FAST_CHECK
    

class ImageExtensions:
    JPG: ClassVar[str] = ".jpg"
    JPEG: ClassVar[str] = ".jpeg"
    PNG: ClassVar[str] = ".png"

    ALL: ClassVar[tuple[str, ...]] = (
        JPG,
        JPEG,
        PNG
    )


    @classmethod
    def list(cls) -> tuple[str, ...]:
        return cls.ALL
    