"""
src/tracking/backends/base.py

Author
------
KrutayaBabka : initial implementation (2026-02-12)

Description
-----------
This module defines the abstract interface (protocol) for tracking backends.

A tracking backend is any implementation that takes video frames as input and
returns a list of tracked objects. This Protocol ensures that all backend
implementations conform to a consistent API, which allows high-level Tracker
classes to use them interchangeably.
"""


from cv2.typing import MatLike
from typing import Protocol, List
from src.tracking.types import TrackedObject


class TrackingBackend(Protocol):
    """
    Abstract interface for a multi-object tracking backend.

    Any concrete tracking backend (YOLO, MediaPipe, etc.) should implement
    this protocol to ensure compatibility with the high-level Tracker class.

    Methods
    -------
    track(frame: MatLike) -> List[TrackedObject]
        Perform object tracking on a single input frame.
    """

    def track(self, frame: MatLike) -> List[TrackedObject]: 
        """
        Track objects in a single video frame.

        Parameters
        ----------
        frame : MatLike
            Input image/frame for which tracking should be performed.

        Returns
        -------
        List[TrackedObject]
            List of tracked objects detected in the frame.

        Notes
        -----
        This is an abstract method and must be implemented by all concrete
        tracking backend classes. The returned list can be empty if no objects
        are detected in the input frame.
        """
        ...
        