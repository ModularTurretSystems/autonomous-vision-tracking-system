"""
examples/aiming_with_calibrated_camera_demo.py

Author: KrutayaBabka
Date: 2026-02-14
Last Modified: 2026-02-14

Description:
    This script demonstrates a complete workflow for:
        1. Loading stereo camera calibration results from an NPZ file.
        2. Initializing an object tracker (YOLO-based).
        3. Selecting a target using a configurable selection strategy.
        4. Computing pan/tilt angles for a turret using camera calibration.
        5. Visualizing tracking, target selection, and aiming direction.

    The demo uses a prerecorded stereo video stream (left camera) and applies
    real-time object tracking and aiming angle computation.

Usage:
    python -m examples.aiming_with_calibrated_camera_demo
"""

import cv2

from src.calibration.persistence.npz import NpzCalibrationStorage
from src.calibration.stereo.types import StereoCalibrationResult
from src.tracking.config import TrackerConfig
from src.tracking.tracker import Tracker
from src.targeting.selector import TargetSelector
from src.targeting.aiming import AimingCalculator

from src.tracking.constants import TrackerModel
from src.targeting.constants import SelectorStrategy

from src.targeting.types import Angles


# -------------------------------
# File and Data Settings
# -------------------------------
DATA_PATH = "data/npz/stereo.npz"
VIDEO_PATH = "data/stereo_videos/left.avi"
IMAGE_SIZE = (1280, 720)

# -------------------------------
# Tracker Settings
# -------------------------------
MODEL = TrackerModel.YOLO26N
CLASSES = None
WITH_REID = True

# -------------------------------
# Aiming Settings
# -------------------------------
PAN_RANGE = (0, 180)
TILT_RANGE = (0, 180)
INITIAL_ANGLES = Angles(pan=90, tilt=90)

# -------------------------------
# Visualization Colors (BGR)
# -------------------------------
COLOR_BOX = (0, 255, 0)
COLOR_TARGET = (0, 0, 255)
COLOR_CENTER = (0, 0, 255)
COLOR_ARROW = (0, 0, 255)
COLOR_TEXT_MAIN = (255, 255, 255)
COLOR_TEXT_META = (255, 0, 0)


# -------------------------------
# Load Calibration Data
# -------------------------------
storage = NpzCalibrationStorage(StereoCalibrationResult)
calibration_data = storage.load(filename=DATA_PATH)

# -------------------------------
# Initialize Tracker
# -------------------------------
tracker_config = TrackerConfig(
    model=MODEL,
    classes=CLASSES,
    device="cuda:0"
)
tracker = Tracker(config=tracker_config)

# -------------------------------
# Target Selection Strategy
# -------------------------------
image_center = (
    calibration_data.left_camera_matrix[0][2],
    calibration_data.left_camera_matrix[1][2]
)
selector = TargetSelector(
    strategy=SelectorStrategy.CLOSEST_TO_CENTER,
    center=image_center
)
image_center = int(image_center[0]), int(image_center[1])

# -------------------------------
# Initialize Aiming Calculator
# -------------------------------
aiming_calculator = AimingCalculator(
    image_size=IMAGE_SIZE,
    camera_matrix=calibration_data.left_camera_matrix,
    pan_range=PAN_RANGE,
    tilt_range=TILT_RANGE
)

# -------------------------------
# Open Video Stream
# -------------------------------
cap = cv2.VideoCapture(
    filename=VIDEO_PATH,
    apiPreference=cv2.CAP_V4L2
)

current_angles = INITIAL_ANGLES

# -------------------------------
# Main Processing Loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    tracked_objects = tracker.track(frame=frame)
    selected_objects = selector.select(objects=tracked_objects)

    if len(selected_objects) == 0:
        continue

    obj = selected_objects[0]

    # Compute aiming angles
    angles = aiming_calculator.compute_pan_tilt_angles(
        current=current_angles,
        target=obj.center
    )

    pt1, pt2, center = obj.pts_to_int()

    # Draw bounding box
    cv2.rectangle(
        img=frame,
        pt1=pt1,
        pt2=pt2,
        color=COLOR_BOX,
        thickness=1
    )

    # Draw target center
    cv2.circle(
        img=frame,
        center=center,
        radius=5,
        color=COLOR_TARGET,
        thickness=-1
    )

    # Draw image center
    cv2.circle(
        img=frame,
        center=image_center,
        radius=5,
        color=COLOR_CENTER,
        thickness=-1
    )

    # Draw aiming direction
    cv2.arrowedLine(
        img=frame,
        pt1=image_center,
        pt2=center,
        color=COLOR_ARROW,
        thickness=2,
        tipLength=0.1
    )

    # Draw angle info
    text = f"pan={round(angles.pan, 2)}, tilt={round(angles.tilt, 2)}"
    cv2.putText(
        img=frame,
        text=text,
        org=(center[0] - 50, center[1] - 20),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.7,
        color=COLOR_TEXT_MAIN,
        thickness=2
    )

    # Draw class info
    text = f"class={obj.cls_name}"
    cv2.putText(
        img=frame,
        text=text,
        org=(center[0] - 50, center[1] + 20),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.7,
        color=COLOR_TEXT_META,
        thickness=2
    )

    # Draw ID info
    text = f"id={obj.id}"
    cv2.putText(
        img=frame,
        text=text,
        org=(center[0] - 50, center[1] + 40),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.7,
        color=COLOR_TEXT_META,
        thickness=2
    )

    cv2.imshow(winname="res", mat=frame)

    if cv2.waitKey(delay=1) == ord("q"):
        break

# -------------------------------
# Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()
