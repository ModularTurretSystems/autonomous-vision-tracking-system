"""
src/tracking/tracker.py

Author
------
KrutayaBabka : initial implementation (2026-02-12)

Description
-----------
This module provides a high-level Tracker interface that wraps a tracking backend.
It allows users to initialize a tracker with a configuration and perform tracking
on video frames. Currently supports YOLO-based backends, with an architecture
ready for extension to other model types.
"""


from .factories.backend_config import BackendConfigFactory
from .backends.yolo import YoloBackend

from .config import TrackerConfig
from .backends.base import TrackingBackend
from .constants import TrackerModel

from typing import List
from cv2.typing import MatLike
from .types import TrackedObject


class Tracker():
    """
    High-level interface for multi-object tracking.

    This class manages the creation of the appropriate backend tracker based on
    the provided configuration and exposes a `track` method for frame-by-frame
    tracking.

    Parameters
    ----------
    config : TrackerConfig
        Configuration object specifying model, task type, thresholds, and other
        tracker parameters.

    Attributes
    ----------
    config : TrackerConfig
        Stores the configuration object.
    model : TrackingBackend
        The backend tracker instance created according to the configuration.
    tracked_objects : List[TrackedObject]
        List of tracked objects from the last processed frame.

    Examples
    --------
    >>> from tracking.config import TrackerConfig
    >>> from tracking.constants import TrackerModel
    >>> config = TrackerConfig(model=TrackerModel.YOLO26N, level=Level.NORMAL)
    >>> tracker = Tracker(config)
    >>> tracked_objects = tracker.track(frame)
    """

    def __init__(
            self, 
            config: TrackerConfig
        ) -> None:
        self.config = config
        self.model = self._create_backend()
        self.tracked_objects: List[TrackedObject] = []

    
    def _create_backend(self) -> TrackingBackend:
        """
        Instantiate the appropriate tracking backend based on the configuration.

        Returns
        -------
        TrackingBackend
            Initialized backend tracker instance.

        Raises
        ------
        ValueError
            If the specified model in the configuration is not supported.

        Notes
        -----
        Currently, only YOLO-based backends are supported. This method is private
        and called automatically during Tracker initialization.
        """

        model = self.config.model

        if model in TrackerModel.YOLO:
            path = BackendConfigFactory.create_yolo_yaml(config=self.config)
            return YoloBackend(config=self.config, tracker=path)
        
        raise ValueError(f"Unknown tracker model: {model}")

    
    def track(self, frame: MatLike) -> List[TrackedObject]:
        """
        Track objects in a single video frame.

        Parameters
        ----------
        frame : MatLike
            Input image/frame to perform tracking on.

        Returns
        -------
        List[TrackedObject]
            List of tracked objects detected in the frame.

        Notes
        -----
        This method delegates tracking to the backend tracker created at
        initialization. The results are cached in `self.tracked_objects`.
        """

        self.tracked_objects = self.model.track(frame=frame)

        return self.tracked_objects
