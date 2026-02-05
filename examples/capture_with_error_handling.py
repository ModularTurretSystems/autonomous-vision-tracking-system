"""
examples/capture_with_error_handling.py

Author: KrutayaBabka
Date: 2026-02-05

Description:
    Example script demonstrating how to capture frames from a single camera using the MonoCapture
    pattern, with robust error handling. The script shows how to gracefully recover from
    camera errors by reinitializing the camera when a capture fails.

    Features:
        - Real-time frame capture and display
        - Exit by pressing a designated key
        - Automatic camera reinitialization on error

Usage:
    python -m examples.capture_with_error_handling
"""

import cv2
import time

from src.camera.camera import Camera


# ===============================
# Constants / Configuration
# ===============================
CAM_ID = 1                    # Camera device index
WIN_NAME = "Mono Capture"     # Window name for display
DELAY = 1                     # Delay for cv2.waitKey (ms)
EXIT_BUTTON = 'q'             # Key to exit capture loop
CAMERA_REINIT_WAIT_SEC = 1.0  # Wait time before retrying camera reinitialization (seconds)


# ===============================
# Initialize Camera
# ===============================
cam = Camera(camera_id=CAM_ID)


# ===============================
# Main Capture Loop
# ===============================
while True:
    try:
        # Capture a frame from the camera
        frame = cam.capture_frame()

    except RuntimeError as e:
        # Handle runtime errors (e.g., camera disconnected or failed)
        print(f"[WARNING] Camera error: {e}")
        print("[INFO] Attempting to reinitialize camera...")
        try:
            cam = Camera(camera_id=CAM_ID)
            print("[INFO] Camera reinitialized successfully.")
        except Exception as init_error:
            print(f"[ERROR] Failed to reinitialize camera: {init_error}")
            time.sleep(CAMERA_REINIT_WAIT_SEC)
        continue  # Retry the loop

    cv2.imshow(winname=WIN_NAME, mat=frame.frame)

    if cv2.waitKey(delay=DELAY) == ord(EXIT_BUTTON):
        break

cv2.destroyAllWindows()
