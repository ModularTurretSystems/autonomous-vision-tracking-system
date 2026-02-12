"""
src/tracking/factories/backend_config.py

Author
------
KrutayaBabka : initial implementation (2026-02-12)

Description
-----------
Factory module for generating backend configuration files for tracking subsystems.

Currently supports creating YAML configuration for YOLO-based backends using
`YoloBackendConfig`. In the future, additional backends (MediaPipe, OpenCV HOG,
etc.) can be supported with their respective factory methods.

This factory ensures that semantic defaults from `TrackerConfig` are applied,
and the resulting configuration is serialized to a YAML file for consumption
by tracking backends.
"""


import yaml
from dataclasses import asdict
from pathlib import Path

from src.tracking.backends.config import YoloBackendConfig
from src.tracking.config import TrackerConfig

from .constants import DEFAULT_IOU, DEFAULT_IMGSZ, OUTPUT_PATH


class BackendConfigFactory:
    """
    Factory for generating backend configuration files.

    Provides static methods to convert high-level `TrackerConfig` instances
    into backend-specific configuration objects and serialize them to YAML.
    """

    @staticmethod
    def create_yolo_yaml(config: TrackerConfig) -> Path:
        """
        Generate a YOLO backend configuration YAML file from a `TrackerConfig`.

        Fills missing threshold values with defaults, maps semantic enumerations
        to concrete numeric values, and serializes the configuration to YAML
        at a predefined path.

        Parameters
        ----------
        config : TrackerConfig
            High-level tracker configuration containing semantic levels,
            thresholds, and model selection.

        Returns
        -------
        Path
            Path object pointing to the saved YAML configuration file.

        Raises
        ------
        OSError
            If the YAML file cannot be written to disk.
        """

        # =========================
        # Ensure detection defaults
        # =========================
        config.iou = config.iou if config.iou is not None else DEFAULT_IOU
        config.imgsz = config.imgsz if config.imgsz is not None else DEFAULT_IMGSZ

        # =========================
        # Create YOLO backend configuration object
        # =========================
        yolo_backend_config = YoloBackendConfig(
            tracker_type      = config.tracker_type.value if config.tracker_type is not None else YoloBackendConfig.tracker_type,
            track_high_thresh = float(config.track_high_thresh) if config.track_high_thresh is not None else YoloBackendConfig.track_high_thresh,
            track_low_thresh  = float(config.track_low_thresh) if config.track_low_thresh is not None else YoloBackendConfig.track_low_thresh,
            new_track_thresh  = float(config.new_track_thresh) if config.new_track_thresh is not None else YoloBackendConfig.new_track_thresh,
            track_buffer      = int(config.track_buffer) if config.track_buffer is not None else YoloBackendConfig.track_buffer,
            match_thresh      = float(config.match_thresh) if config.match_thresh is not None else YoloBackendConfig.match_thresh,
            fuse_score        = config.fuse_score if config.fuse_score is not None else YoloBackendConfig.fuse_score,
            gmc_method        = config.gmc_method.value if config.gmc_method is not None else YoloBackendConfig.gmc_method,
            proximity_thresh  = float(config.proximity_thresh) if config.proximity_thresh is not None else YoloBackendConfig.proximity_thresh,
            appearance_thresh = float(config.appearance_thresh) if config.appearance_thresh is not None else YoloBackendConfig.appearance_thresh,
            with_reid         = config.with_reid if config.with_reid is not None else YoloBackendConfig.with_reid,
            model             = YoloBackendConfig.model
        )

        # =========================
        # Serialize configuration to YAML
        # =========================
        path = OUTPUT_PATH
        path.parent.mkdir(parents=True, exist_ok=True) # ensure directories exist
        with open(file=path, mode="w") as f:
            yaml.safe_dump(data=asdict(yolo_backend_config), stream=f, sort_keys=False)

        return path
