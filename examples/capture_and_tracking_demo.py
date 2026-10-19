"""
examples/capture_and_tracking_demo.py

Author: KrutayaBabka
Date: 2026-02-14
Last Modified: 2026-02-14

Description:
    This script demonstrates a complete workflow for:
        1. Capturing frames from a camera device.
        2. Running an object detection/tracking model (YOLO-based).
        3. Visualizing detected and tracked objects with bounding boxes and labels.

    The demo uses a live camera stream and applies real-time object tracking.

Usage:
    python -m examples.capture_and_tracking_demo
"""


import cv2

from src.tracking.tracker import Tracker
from src.tracking.config import TrackerConfig
from src.camera.camera import Camera

from src.tracking.constants import TrackerModel, TaskType


# -------------------------------
# Tracker Settings
# -------------------------------
MODEL = TrackerModel.YOLO26N
TASK = TaskType.DETECT
CLASSES = 0
TRACK_BUFFER = 300
WITH_REID = True
DEVICE = "cuda:0"
CONF = 0.5
HALF = True

# -------------------------------
# Camera Settings
# -------------------------------
CAM_ID = 0
API_PREFERENCE = cv2.CAP_V4L2
CAM_PARAMS = [
    cv2.CAP_PROP_FRAME_WIDTH, 1280,
    cv2.CAP_PROP_FRAME_HEIGHT, 720
]

# -------------------------------
# Visualization Settings
# -------------------------------
DELAY = 1
EXIT_BUTTON = "q"

COLOR_BOX = (0, 255, 0)
COLOR_TEXT = (0, 255, 0)


# -------------------------------
# Initialize Tracker
# -------------------------------
tracker_config = TrackerConfig(
    model=MODEL,
    task=TASK,
    classes=CLASSES,
    track_buffer=TRACK_BUFFER,
    with_reid=WITH_REID,
    device=DEVICE,
    conf=CONF,
    half=HALF
)
tracker = Tracker(config=tracker_config)

# -------------------------------
# Initialize Camera
# -------------------------------
camera = Camera(
    camera_id=CAM_ID,
    apiPreference=API_PREFERENCE,
    params=CAM_PARAMS
)

# -------------------------------
# Main Processing Loop
# -------------------------------
while True:
    frame = camera.capture_frame()

    result = tracker.track(frame=frame.frame)

    for obj in result.objects:
        obj_id = obj.id
        class_name = obj.cls_name

        pt1, pt2 = obj.box.pts_to_int()

        # Draw bounding box
        cv2.rectangle(
            img=frame.frame,
            pt1=pt1,
            pt2=pt2,
            color=COLOR_BOX,
            thickness=2
        )

        # Draw label
        cv2.putText(
            img=frame.frame,
            text=f"ID:{obj_id}, Class:{class_name}",
            org=(pt1[0], pt1[1] - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=COLOR_TEXT,
            thickness=1
        )

    cv2.imshow(winname="res", mat=frame.frame)

    if cv2.waitKey(delay=DELAY) == ord(EXIT_BUTTON):
        break

# -------------------------------
# Cleanup
# -------------------------------
cv2.destroyAllWindows()
