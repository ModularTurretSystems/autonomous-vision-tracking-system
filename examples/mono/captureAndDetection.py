import cv2

from src.camera.camera import Camera
from src.vision.mono.capture import MonoCapture

# ========== CONSTANTS ==========
CAM_ID = 0
SAVE_DIR = "frames"

WIN_NAME = "mono"

DELAY = 1

EXIT_BUTTON = 'q'
SAVE_BUTTON = 's'

BASE_NAME = "frame"
EXT = "jpg"

SAVE_MESSAGE = "Saved"

# ==================================================


cam = Camera(camera_id=CAM_ID)

capture = MonoCapture(camera=cam, save_dir=SAVE_DIR)
# capture = MonoCapture(camera=0) # Another method to init MonoCapture

while(1):
    frame = capture.capture_frame()
    
    cv2.imshow(winname=WIN_NAME, mat=frame)

    k = cv2.waitKey(delay=DELAY)

    if k == ord(EXIT_BUTTON):
        break
    elif k == ord(SAVE_BUTTON):
        if capture.save_frame(frame=frame, base_name=BASE_NAME, ext=EXT): print(SAVE_MESSAGE)

print(f"Number of saved frames = {capture.get_saved_frame_count()}")
