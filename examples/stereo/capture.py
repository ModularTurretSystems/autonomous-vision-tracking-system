import cv2
import os

from src.camera.camera import Camera
from src.vision.stereo.system import StereoSystem
from src.vision.stereo.capture import StereoCapture
from src.calibration.patterns.chessboard import ChessboardPattern

from src.utils.image import combine_and_resize_frames


os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

PARAMS = (
    cv2.CAP_PROP_FRAME_WIDTH, 1280,
    cv2.CAP_PROP_FRAME_HEIGHT, 720,
)
SAVE_DIR = "data/calibration_images"

PATTERN_SIZE = (9, 6)
FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
WIN_SIZE = (11, 11)
ZERO_ZONE = (-1, -1)
CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

COMBINED_RESOLUTION = (1280, 720)


cam_l = Camera(camera_id=1, apiPreference=cv2.CAP_MSMF, params=PARAMS)
cam_r = Camera(camera_id=2, apiPreference=cv2.CAP_MSMF, params=PARAMS)
stereo_system = StereoSystem(left=cam_l, right=cam_r)
stereo_capture = StereoCapture(stereo_system=stereo_system, save_dir=SAVE_DIR)

pattern = ChessboardPattern(pattern_size=PATTERN_SIZE, flags=FLAGS, refine=True, win_size=WIN_SIZE, zero_zone=ZERO_ZONE, criteria=CRITERIA)


while(1):
    stereo_frame = stereo_capture.capture_frame()
    frame_l = stereo_frame.camera_frame_l.frame
    frame_r = stereo_frame.camera_frame_r.frame

    res_l = pattern.detect_corners(img=frame_l)
    res_r = pattern.detect_corners(img=frame_r)

    stereo_frame_copy = stereo_frame.copy()

    pattern.draw_corners(img=stereo_frame_copy.camera_frame_l.frame, corners=res_l.corners, patternWasFound=res_l.found)
    pattern.draw_corners(img=stereo_frame_copy.camera_frame_r.frame, corners=res_r.corners, patternWasFound=res_r.found)

    stereo_frame_copy.flip(flip_code=1)
    frame = combine_and_resize_frames(frame_size=COMBINED_RESOLUTION, frames=stereo_frame_copy.to_list(), horizontal=True)

    cv2.imshow("c", frame)

    k = cv2.waitKey(delay=1)

    if k == ord('q'):
        break
    
    if k == ord('s') and res_l.found and res_r.found:
        stereo_capture.save_frame(frame=stereo_frame)
