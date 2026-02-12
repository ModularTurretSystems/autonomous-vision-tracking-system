"""
src/tracking/constants.py

Author
------
KrutayaBabka : initial implementation (2026-02-12)

Description
-----------
This module defines semantic constants and enumerations used across the
tracking subsystem.

It provides:
- Supported YOLO model identifiers
- Tracker and task types
- Predefined threshold levels for detection and tracking logic
- Motion compensation (GMC) method identifiers

All constants are expressed as strongly-typed enumerations in order to:
- Avoid magic numbers and magic strings
- Provide semantic meaning to configuration values
- Improve IDE autocompletion and static type checking
"""


from enum import StrEnum, Enum, IntEnum
from typing import List


# =========================
# Model and Task Identifiers
# =========================

class YoloModels:
    """
    Supported YOLO model identifiers.
    """

    YOLO26N     = "yolo26n"
    YOLOV8N_SEG = "yolo8n-seg"  


class TrackerModel(YoloModels):
    """
    Collection of model identifiers for tracking purposes.

    Inherits from `YoloModels` to retain existing YOLO constants.
    Provides a class-level list `YOLO` for convenience. 

    Future model sets (MediaPipe, cv2, etc.) can be added similarly.
    """
    
    YOLO: List[str] = [v for k, v in YoloModels.__dict__.items() if not k.startswith("__") or not k.startswith("_")]


class TrackerType(StrEnum):
    """
    Supported multi-object tracking algorithms.
    """

    BOTSORT   = "botsort"
    BYTETRACK = "bytetrack"


class TaskType(StrEnum):
    """
    Supported inference task types.
    """

    DETECT   = "detect"
    SEGMENT  = "segment"
    CLASSIFY = "classify"
    POSE     = "pose"
    OBB      = "obb"


# =========================
# Detection Thresholds
# =========================

class IoUThresh(float, Enum):
    """
    Intersection-over-Union (IoU) thresholds for detection association.
    """

    VERY_LOW  = 0.30
    LOW       = 0.40
    NORMAL    = 0.50
    HIGH      = 0.60
    VERY_HIGH = 0.70


class ConfThresh(float, Enum):
    """
    Confidence score thresholds for object detection filtering.
    """

    VERY_LOW  = 0.15
    LOW       = 0.25
    NORMAL    = 0.35
    HIGH      = 0.45
    VERY_HIGH = 0.60


# =========================
# Tracking Thresholds
# =========================

class TrackHighThresh(float, Enum):
    """
    High confidence threshold for track continuation.
    """

    VERY_LOW  = 0.35
    LOW       = 0.4
    NORMAL    = 0.5
    HIGH      = 0.6
    VERY_HIGH = 0.7


class TrackLowThresh(float, Enum):
    """
    Low confidence threshold for tentative track handling.
    """

    VERY_LOW  = 0.05
    LOW       = 0.1
    NORMAL    = 0.2
    HIGH      = 0.3
    VERY_HIGH = 0.4


class NewTrackThresh(float, Enum):
    """
    Threshold for initializing new tracks.
    """

    VERY_LOW  = 0.45
    LOW       = 0.5
    NORMAL    = 0.6
    HIGH      = 0.7
    VERY_HIGH = 0.8


class TrackBuffer(IntEnum):
    """
    Number of frames a track is kept alive without updates.
    """

    VERY_LOW  = 10
    LOW       = 15
    NORMAL    = 30
    HIGH      = 45
    VERY_HIGH = 60


class MatchThresh(float, Enum):
    """
    Threshold for matching detections to existing tracks.
    """

    VERY_LOW  = 0.65
    LOW       = 0.7
    NORMAL    = 0.8
    HIGH      = 0.85
    VERY_HIGH = 0.9


# =========================
# Motion Compensation
# =========================

class GMCMethod(StrEnum):
    """
    Global Motion Compensation (GMC) method identifiers.
    """

    NONE            = "none"
    ORB             = "orb"
    SIFT            = "sift"
    ECC             = "ecc"
    SPARSE_OPT_FLOW = "sparseOptFlow"


# =========================
# Re-identification Thresholds
# =========================

class ProximityThresh(float, Enum):
    """
    Spatial proximity threshold for track association.
    """

    VERY_LOW  = 0.35
    LOW       = 0.4
    NORMAL    = 0.5
    HIGH      = 0.6
    VERY_HIGH = 0.7


class AppearanceThresh(float, Enum):
    """
    Appearance embedding similarity threshold for re-identification.
    """

    VERY_LOW  = 0.2
    LOW       = 0.25
    NORMAL    = 0.35
    HIGH      = 0.45
    VERY_HIGH = 0.55


# =========================
# Generic Levels
# =========================

class Level(IntEnum):
    """
    Generic discrete level enumeration.

    Can be used to index or map semantic levels to thresholds.
    """

    VERY_LOW  = 0
    LOW       = 1
    NORMAL    = 2
    HIGH      = 3
    VERY_HIGH = 4
