"""
src/tracking/backends/config.py

Author
------
KrutayaBabka : initial implementation (2026-02-12)

Description
-----------
Backend-specific configuration for multi-object trackers.

Currently includes configuration for YOLO-based tracking backends. In the
future, additional backends (MediaPipe, OpenCV HOG, etc.) can be added with
their own configuration classes.

These configuration objects are intended to be serialized to YAML files via
`BackendConfigFactory` for use with the tracking backends.
"""


from dataclasses import dataclass


@dataclass
class YoloBackendConfig:
    """
    Configuration parameters for a YOLO-based tracking backend.

    This class encapsulates all hyperparameters controlling the tracking
    algorithm behavior, motion compensation, and optional re-identification.

    Parameters
    ----------
    tracker_type : str
        Type of tracking algorithm to use, e.g., "botsort" or "bytetrack".
    track_high_thresh : float
        High confidence threshold for maintaining tracks.
    track_low_thresh : float
        Low confidence threshold for tentative tracks.
    new_track_thresh : float
        Threshold for initializing new tracks.
    track_buffer : int
        Maximum number of frames a track can persist without updates.
    match_thresh : float
        Threshold for matching new detections to existing tracks.
    fuse_score : bool
        Whether to fuse detection and track scores.
    gmc_method : str
        Global Motion Compensation (GMC) method, e.g., "sparseOptFlow".
    proximity_thresh : float
        Maximum spatial distance for associating tracks.
    appearance_thresh : float
        Threshold for appearance similarity in re-identification.
    with_reid : bool
        Enable appearance-based re-identification.
    model : str
        Model selection strategy or path; "auto" for automatic selection.
    """

    # =========================
    # Tracking thresholds
    # =========================
    tracker_type: str = "botsort"
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True


    # =========================
    # Motion Compensation (GMC)
    # =========================
    gmc_method: str = "sparseOptFlow"


    # =========================
    # Re-identification thresholds
    # =========================
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.8
    with_reid: bool = False


    # =========================
    # Model selection
    # =========================
    model: str = "auto"
