from src.calibration.patterns.chessboard import ChessboardPattern
from src.utils.image import combine_and_resize_frames
from src.camera.camera import Camera
import cv2 

import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

params = (
    cv2.CAP_PROP_FRAME_WIDTH, 1280,
    cv2.CAP_PROP_FRAME_HEIGHT, 720,
)

cv2.TERM_CRITERIA_COUNT

PATTERN_SIZE = (9, 6)
WIN_SIZE = (11, 11)
ZERO_ZONE = (-1, -1)
CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.01
)

cam = Camera(camera_id=0, apiPreference=cv2.CAP_MSMF, params=params)

CAM_RESOLUTION = cam.get_cam_resolution()
SCREEN_RESOLUTION = (1280, 720)

ROWS, COLS = 2, 2

img_w = SCREEN_RESOLUTION[0] // COLS
img_h = SCREEN_RESOLUTION[1] // ROWS

FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS

pattern = ChessboardPattern(pattern_size=PATTERN_SIZE, flags=FLAGS, refine=True, win_size=WIN_SIZE, zero_zone=ZERO_ZONE, criteria=CRITERIA)

while(1):
    _, frame = cam.capture_frame()

    res = pattern.detect_corners(img=frame)
    
    frame_with_corners = frame.copy()
    pattern.draw_corners(img=frame_with_corners, corners=res.corners, patternWasFound=res.found)
    cv2.flip(src=frame_with_corners, flipCode=1, dst=frame_with_corners)

    res_frame = combine_and_resize_frames(frame_size=SCREEN_RESOLUTION, frames=[frame, frame_with_corners], horizontal=True)

    cv2.imshow(winname="Corners", mat=res_frame)

    if cv2.waitKey(delay=1) == ord('q'):
        break

cv2.destroyAllWindows()
