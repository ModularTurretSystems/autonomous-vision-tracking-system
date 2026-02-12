"""
src/tracking/config.py

Author
------
KrutayaBabka : initial implementation (2026-02-12)

Description
-----------
This module defines the `TrackerConfig` dataclass used to configure multi-object
tracking (MOT) pipelines. It encapsulates model selection, task type, detection
and tracking thresholds, tracking buffer settings, GMC methods, and other runtime
parameters.

The class supports semantic preset levels (`Level`) that automatically fill
thresholds for IoU, confidence, and tracking metrics if not explicitly provided.
"""


from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

from .constants import (
    TrackerModel,
    TaskType, 
    TrackerType,
    IoUThresh,
    ConfThresh,
    TrackHighThresh,
    TrackLowThresh,
    NewTrackThresh,
    TrackBuffer,
    MatchThresh,
    GMCMethod,
    ProximityThresh,
    AppearanceThresh,
    Level
)


@dataclass
class TrackerConfig:
    """
    Configuration container for a multi-object tracker instance.

    This dataclass stores all parameters required to configure a tracker,
    including model selection, detection thresholds, inference options, and
    tracking-specific thresholds. If a semantic `level` is provided, thresholds
    are automatically filled according to that level unless explicitly set.

    Parameters
    ----------
    model : TrackerModel
        Identifier of the model backend to be used (YOLO, MediaPipe, etc.).
    task : TaskType, optional
        Type of inference task, e.g., detection, segmentation, pose estimation.
    level : Level, optional
        Semantic preset level that defines default thresholds. Overrides unspecified
        numeric thresholds if provided.
    classes : int or list of int, optional
        Specific class indices to track. `None` means all detected classes.
    stream : bool
        Whether the input source is a video stream. Default is True.
    persist : bool
        Whether to keep resources (e.g., model weights) loaded across multiple runs.
        Default is True.
    verbose : bool
        Enable verbose logging. Default is False.

    iou : IoUThresh or float, optional
        Intersection-over-Union threshold for detection association.
    conf : ConfThresh or float, optional
        Confidence threshold for filtering detections.
    imgsz : int or tuple of int, optional
        Image size to be used for inference. Can be provided as a single integer
        (applied to both height and width) or as a tuple `(height, width)`.
        If `None`, default model image size will be used.

    half : bool
        Use half-precision (FP16) inference if supported. Default is False.
    rect : bool
        Use rectangular inference (optimized cropping). Default is True.
    augment : bool
        Apply augmentation during inference. Default is False.
    agnostic_nms : bool
        Class-agnostic non-max suppression. Default is False.
    retina_masks : bool
        Use high-resolution masks for segmentation. Default is False.
    device : str, optional
        Device identifier for inference (e.g., "cpu", "cuda:0").

    tracker_type : TrackerType, optional
        Tracking algorithm type (e.g., ByteTrack, BoTSORT).
    track_high_thresh : TrackHighThresh or float, optional
        High confidence threshold for maintaining tracks.
    track_low_thresh : TrackLowThresh or float, optional
        Low confidence threshold for tentative tracks.
    new_track_thresh : NewTrackThresh or float, optional
        Threshold for initializing new tracks.
    track_buffer : TrackBuffer or int, optional
        Maximum number of frames a track can persist without updates.
    match_thresh : MatchThresh or float, optional
        Threshold for matching new detections to existing tracks.
    fuse_score : bool, optional
        Whether to fuse detection and track scores.
    gmc_method : GMCMethod, optional
        Global Motion Compensation method to stabilize tracks.
    proximity_thresh : ProximityThresh or float, optional
        Maximum spatial distance for associating tracks.
    appearance_thresh : AppearanceThresh or float, optional
        Threshold for appearance similarity in re-identification.
    with_reid : bool, optional
        Enable appearance-based re-identification.

    Notes
    -----
    Threshold fields (`iou`, `conf`, `track_high_thresh`, etc.) are automatically
    filled based on `level` if not explicitly provided. This allows convenient
    preset configurations:
    
    - Level.VERY_LOW
    - Level.LOW
    - Level.NORMAL
    - Level.HIGH
    - Level.VERY_HIGH

    Examples
    --------
    Create a tracker config with semantic level NORMAL:

    >>> config = TrackerConfig(model=TrackerModel.YOLO26N, level=Level.NORMAL)
    >>> config.iou
    0.5
    >>> config.track_high_thresh
    0.5
    """

    # =========================
    # Core model and task
    # =========================
    model:             Union[TrackerModel, str]
    task:              Optional[TaskType]                       = None
    level:             Optional[Level]                          = None
    classes:           Optional[Union[int, List[int]]]          = None
    stream:            bool                                     = True
    persist:           bool                                     = True
    verbose:           bool                                     = False


    # =========================
    # Detection thresholds
    # =========================
    iou:               Optional[Union[IoUThresh, float]]        = None
    conf:              Optional[Union[ConfThresh, float]]       = None
    imgsz:             Optional[Union[int, Tuple[int, int]]]    = None


    # =========================
    # Inference options
    # =========================
    half:              bool                                     = False
    rect:              bool                                     = True
    augment:           bool                                     = False
    agnostic_nms:      bool                                     = False
    retina_masks:      bool                                     = False
    device:            Optional[str]                            = None


    # =========================
    # Tracking parameters
    # =========================
    tracker_type:      Optional[TrackerType]                    = None
    track_high_thresh: Optional[Union[TrackHighThresh, float]]  = None
    track_low_thresh:  Optional[Union[TrackLowThresh, float]]   = None
    new_track_thresh:  Optional[Union[NewTrackThresh, float]]   = None
    track_buffer:      Optional[Union[TrackBuffer, int]]        = None
    match_thresh:      Optional[Union[MatchThresh, float]]      = None
    fuse_score:        Optional[bool]                           = None
    gmc_method:        Optional[GMCMethod]                      = None
    proximity_thresh:  Optional[Union[ProximityThresh, float]]  = None
    appearance_thresh: Optional[Union[AppearanceThresh, float]] = None
    with_reid:         Optional[bool]                           = None


    def __post_init__(self):
        """
        Automatically fill threshold values from the semantic `level` if not explicitly provided.

        This method is called automatically after the dataclass is initialized. 
        If a `level` is set, default values for detection and tracking thresholds are automatically
        filled based on the selected `level`. This avoids having to manually set all individual thresholds.

        Notes
        -----
        The following thresholds are set according to `level` if they were not explicitly provided:
        
        - `iou` -> IoUThresh[level]
        - `conf` -> ConfThresh[level]
        - `track_high_thresh` -> TrackHighThresh[level]
        - `track_low_thresh` -> TrackLowThresh[level]
        - `new_track_thresh` -> NewTrackThresh[level]
        - `track_buffer` -> TrackBuffer[level]
        - `match_thresh` -> MatchThresh[level]
        - `proximity_thresh` -> ProximityThresh[level]
        - `appearance_thresh` -> AppearanceThresh[level]

        Example
        -------
        Automatically filled thresholds after object creation:

        >>> config = TrackerConfig(model=TrackerModel.YOLO26N, level=Level.NORMAL)
        >>> config.iou
        0.5
        >>> config.track_high_thresh
        0.5
        """

        if self.level is not None:
            self.iou                 = self.iou                 or IoUThresh[self.level.name]
            self.conf                = self.conf                or ConfThresh[self.level.name]
            self.track_high_thresh   = self.track_high_thresh   or TrackHighThresh[self.level.name]
            self.track_low_thresh    = self.track_low_thresh    or TrackLowThresh[self.level.name]
            self.new_track_thresh    = self.new_track_thresh    or NewTrackThresh[self.level.name]
            self.track_buffer        = self.track_buffer        or TrackBuffer[self.level.name]
            self.match_thresh        = self.match_thresh        or MatchThresh[self.level.name]
            self.proximity_thresh    = self.proximity_thresh    or ProximityThresh[self.level.name]
            self.appearance_thresh   = self.appearance_thresh   or AppearanceThresh[self.level.name]
