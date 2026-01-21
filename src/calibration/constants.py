from enum import IntEnum
import cv2

class ChessboardFlags(IntEnum):
    DEFAULT = 0
    ADAPTIVE_THRESH = cv2.CALIB_CB_ADAPTIVE_THRESH
    NORMALIZE_IMAGE = cv2.CALIB_CB_NORMALIZE_IMAGE
    FAST_CHECK = cv2.CALIB_CB_FAST_CHECK
    