"""
src/tracking/factories/constants.py

Author
------
KrutayaBabka : initial implementation (2026-02-12)

Description
-----------
Module defining constants for the backend configuration factories.

Provides default paths and parameters used when generating configuration
files for tracker backends, e.g., YOLO.
"""


from pathlib import Path


# =========================
# Output paths
# =========================
OUTPUT_PATH = Path("data/configs/yolo_config.yaml")
"""
Path to the YAML configuration file generated for YOLO backend.

This file will store serialized parameters such as track thresholds,
motion compensation settings, and model identifiers. The parent
directory will be automatically created if it does not exist.
"""


# =========================
# Default detection thresholds
# =========================
DEFAULT_IOU = 0.7
"""
Default Intersection-over-Union (IoU) threshold for associating
detections with existing tracks if not specified in `TrackerConfig`.
"""

DEFAULT_IMGSZ = 640
"""
Default image size used for inference when none is specified.

Applied as both height and width for square inputs. Matches
common YOLO model training resolutions.
"""
