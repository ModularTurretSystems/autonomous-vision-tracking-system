"""
src/tracking/backends/base.py

Author
------
KrutayaBabka : initial implementation (2026-02-12)

Description
-----------
This module defines the abstract interface (protocol) for tracking backends.

A tracking backend is any implementation that takes video frames as input and
returns a structured `TrackedObjects` container. This Protocol ensures that
all backend implementations conform to a consistent API, allowing high-level
Tracker classes to use them interchangeably.
"""


from cv2.typing import MatLike
from typing import Protocol
from src.tracking.types import TrackedObjects


class TrackingBackend(Protocol):
    """
    Abstract interface for a multi-object tracking backend.

    Any concrete tracking backend (YOLO, MediaPipe, etc.) should implement
    this protocol to ensure compatibility with the high-level `Tracker` class.

    Methods
    -------
    track(frame: MatLike) -> TrackedObjects
        Perform object tracking on a single input frame.
    """

    def track(self, frame: MatLike) -> TrackedObjects: 
        """
        Track objects in a single video frame.

        Parameters
        ----------
        frame : MatLike
            Input image/frame on which tracking should be performed.

        Returns
        -------
        TrackedObjects
            Structured container of detected/tracked objects for the frame.
            Each `Object` in `TrackedObjects.objects` contains:
            - `cls_id` : int, numeric class ID
            - `cls_name` : str, human-readable class name
            - `conf` : float, detection/tracking confidence
            - `id` : int | None, tracking ID if available
            - `box` : Box, bounding box coordinates (pt1, pt2)
            `orig_shape` contains the original frame height and width.
            `raw_data` may contain raw outputs from the model if available.

        Notes
        -----
        This is an abstract method and must be implemented by all concrete
        tracking backend classes. The returned `TrackedObjects` can have
        an empty `objects` list if no objects are detected in the input frame.
        """
        ...
        