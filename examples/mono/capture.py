"""
examples/mono/capture.py

Author: KrutayaBabka
Date: 2026-01-24
Description: 
    Example script demonstrating how to capture and save frames from a single camera
    using the MonoCapture class. Frames are displayed in a window and can be saved
    by pressing a designated key. 

Usage:
    python -m examples.mono.capture
"""


import cv2

from src.camera.camera import Camera
from src.vision.mono.capture import MonoCapture


# ========== CONSTANTS ==========

# Camera ID used to select the physical camera device.
# Can be an integer index (0, 1, 2...) depending on connected cameras.
CAM_ID = 0

# Directory where captured frames will be saved.
# Can be relative or absolute path.
SAVE_DIR = "frames"

# Name of the OpenCV window for displaying live frames.
WIN_NAME = "mono"

# Delay in milliseconds for cv2.waitKey().
# Controls refresh rate of the display and responsiveness to key presses.
DELAY = 1

# Key used to exit the capture loop.
EXIT_BUTTON = 'q'

# Key used to save the current frame to disk.
SAVE_BUTTON = 's'

# Base filename for saved frames.
# Each frame will be saved as BASE_NAME + incremented number + extension.
BASE_NAME = "frame"

# File extension/format for saved frames.
# Can be 'jpg', 'png', etc.
EXT = "jpg"

# Message printed to the console when a frame is successfully saved.
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
