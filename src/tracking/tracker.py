"""
src/tracking/tracker.py

Author
------
KrutayaBabka : initial implementation (2026-02-12)

Description
-----------
High-level Tracker interface for multi-object tracking. 

This module wraps backend tracker implementations (currently YOLO-based)
and exposes a unified `Tracker` class for frame-by-frame tracking. Results
are returned as structured `TrackedObjects`, containing lightweight `Object`
instances with bounding boxes (`Box`), class information, confidence scores,
and tracking IDs.
"""


from .factories.backend_config import BackendConfigFactory
from .backends.yolo import YoloBackend

from .config import TrackerConfig
from .backends.base import TrackingBackend
from .constants import TrackerModel

from cv2.typing import MatLike
from .types import TrackedObjects


class Tracker():
    """
    High-level multi-object tracker interface.

    This class handles backend selection and initialization based on
    `TrackerConfig`. Users can perform tracking on video frames using
    the `track` method, receiving results in a consistent internal format.

    Parameters
    ----------
    config : TrackerConfig
        Tracker configuration specifying model type, task, thresholds,
        inference options, and tracking parameters.

    Attributes
    ----------
    config : TrackerConfig
        The configuration used to initialize the tracker.
    model : TrackingBackend
        The instantiated backend tracker according to the configuration.

    Examples
    --------
    >>> from tracking.config import TrackerConfig
    >>> from tracking.constants import TrackerModel, Level
    >>> from tracking.tracker import Tracker
    >>> config = TrackerConfig(model=TrackerModel.YOLO26N, level=Level.NORMAL)
    >>> tracker = Tracker(config)
    >>> tracked_objects = tracker.track(frame)
    >>> print(tracked_objects.objects[0].box.pt1)  # Top-left corner of first detected object
    """

    def __init__(
            self, 
            config: TrackerConfig
        ) -> None:
        self.config = config
        self.model = self._create_backend()

    
    def _create_backend(self) -> TrackingBackend:
        """
        Instantiate the backend tracker according to the configuration.

        Returns
        -------
        TrackingBackend
            The initialized backend tracker instance.

        Raises
        ------
        ValueError
            If the specified model in the configuration is unsupported.

        Notes
        -----
        Currently only YOLO-based backends are supported. This method is
        private and called automatically during Tracker initialization.
        """

        model = self.config.model

        if model in TrackerModel.YOLO:
            path = BackendConfigFactory.create_yolo_yaml(config=self.config)
            return YoloBackend(config=self.config, tracker=path)
        
        raise ValueError(f"Unknown tracker model: {model}")

    
    def track(self, frame: MatLike) -> TrackedObjects:
        """
        Perform detection and tracking on a single video frame.

        Parameters
        ----------
        frame : MatLike
            Input image or video frame to process.

        Returns
        -------
        TrackedObjects
            Structured container of detected/tracked objects for the frame.
            Each `Object` includes:
            - `cls_id` : int, numeric class ID
            - `cls_name` : str, human-readable class name
            - `conf` : float, detection/tracking confidence
            - `id` : int or None, tracking identifier if available
            - `box` : Box, bounding box in pixel coordinates (pt1, pt2)

        Notes
        -----
        - Delegates tracking to the backend tracker created during initialization.
        - No internal caching of results is performed; the method returns
          results directly in `TrackedObjects`.
        """

        return self.model.track(frame=frame)
