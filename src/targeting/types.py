"""
src/targeting/types.py

Author
------
KrutayaBabka : initial implementation (2026-02-14)

Description
-----------
Defines data structures used for target selection and aiming logic.

This module extends the generic tracked object representation from `src.tracking.types`
and provides additional information needed for selecting a target and computing
angles for pan-tilt systems.
"""


from dataclasses import dataclass
from cv2.typing import Point2f, Point
from src.tracking.types import Object
from typing import Tuple


@dataclass(slots=True)
class Angles:
    """
    Represents pan and tilt angles for a target.

    Parameters
    ----------
    pan : float
        Horizontal rotation angle (degrees).
    tilt : float
        Vertical rotation angle (degrees).

    Notes
    -----
    This class is intended to store computed aiming angles for a turret,
    camera, or other pan-tilt system.
    """

    pan: float
    tilt: float


@dataclass(slots=True)
class SelectedObject(Object):
    """
    Extended object representation for target selection.

    Inherits from `Object` and adds metrics and center coordinates
    used to prioritize and aim at a target.

    Parameters
    ----------
    cls_id : int
        Numeric class identifier (inherited from Object).
    cls_name : str
        Human-readable class name (inherited from Object).
    conf : float
        Detection/tracking confidence (inherited from Object).
    id : int or None
        Tracking ID (inherited from Object).
    box : Box
        Bounding box of the object (inherited from Object).
    metric_raw : float
        Raw score for target prioritization (e.g., distance, area).
    metric_norm : float
        Normalized score for target prioritization (0-1 range).
    center : Point2f
        Center point of the target in pixel coordinates.

    Notes
    -----
    This class is intended to represent a candidate target with additional
    attributes required for selection algorithms, aiming, and control logic.
    """

    metric_raw: float
    metric_norm: float
    center: Point2f


    @classmethod
    def from_Object(cls, object: Object, center: Point2f, metric_raw: float, metric_norm: float) -> "SelectedObject":
        """
        Create a `SelectedObject` from a generic `Object` instance.

        Parameters
        ----------
        obj : Object
            Base tracked object.
        center : Point2f
            Center of the target in pixel coordinates.
        metric_raw : float
            Raw prioritization score for the target.
        metric_norm : float
            Normalized prioritization score (0-1 range).

        Returns
        -------
        SelectedObject
            New instance with extended targeting attributes.
        """

        return cls(
            cls_id=object.cls_id,
            cls_name=object.cls_name,
            conf=object.conf,
            id=object.id,
            box=object.box,
            metric_raw = metric_raw,
            metric_norm=metric_norm,
            center = center
        )


    def center_to_int(self) -> Point:
        """
        Convert the floating-point center coordinates to integers.

        Returns
        -------
        Point
            Tuple (x, y) of the center as integer pixel coordinates.
        """

        return (int(self.center[0]), int(self.center[1]))
    

    def pts_to_int(self) -> Tuple[Point, Point, Point]:
        """
        Convert bounding box corners and center point to integer coordinates.

        Returns
        -------
        tuple of Point
            Tuple containing (top-left, bottom-right, center) coordinates
            as integer pixel values.
        """
        
        return (*self.box.pts_to_int(), self.center_to_int())
