import cv2

from src.calibration.patterns.chessboard import ChessboardPattern
from src.camera.camera import Camera
from src.vision.mono.capture import MonoCapture

from src.utils.image import combine_and_resize_frames


# ========== CONSTANTS ==========
CAM_ID = 0
SAVE_DIR = "data/frames"

WIN_NAME = "Capture & Detection"

DELAY = 1

EXIT_BUTTON = 'q'
SAVE_BUTTON = 's'

BASE_NAME = "frame"
EXT = "jpg"

SAVE_MESSAGE = "Saved"

PARAMS = (
    cv2.CAP_PROP_FRAME_WIDTH, 1280,
    cv2.CAP_PROP_FRAME_HEIGHT, 720,
)

PATTERN_SIZE = (9, 6)
WIN_SIZE = (11, 11)
ZERO_ZONE = (-1, -1)
CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.01
)

COMBINED_FRAME_SIZE = (1280, 720)

FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS

# ==================================================


cam = Camera(camera_id=CAM_ID, apiPreference=cv2.CAP_MSMF, params=PARAMS)
pattern = ChessboardPattern(pattern_size=PATTERN_SIZE, flags=FLAGS, refine=True, win_size=WIN_SIZE, zero_zone=ZERO_ZONE, criteria=CRITERIA)

capture = MonoCapture(camera=cam, save_dir=SAVE_DIR)
# capture = MonoCapture(camera=0) # Another method to init MonoCapture


while(1):
    camera_frame = capture.capture_frame()

    res = pattern.detect_corners(camera_frame.frame)
    camera_frame_copy = camera_frame.copy()
    pattern.draw_corners(img=camera_frame_copy.frame, corners=res.corners, patternWasFound=res.found)
    
    frame = combine_and_resize_frames(frame_size=COMBINED_FRAME_SIZE, frames=[camera_frame.flip(flip_code=1, in_place=False), camera_frame_copy.flip(flip_code=1)])

    cv2.imshow(winname=WIN_NAME, mat=frame)

    k = cv2.waitKey(delay=DELAY)

    if k == ord(EXIT_BUTTON):
        break
    elif k == ord(SAVE_BUTTON):
        if capture.save_frame(frame=camera_frame.frame, base_name=BASE_NAME, ext=EXT): print(SAVE_MESSAGE)

print(f"Number of saved frames = {capture.get_saved_frame_count()}")
