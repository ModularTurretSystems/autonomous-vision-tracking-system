"""
src/tracking/types.py

Author
------
dolby228 : original implementation (2026-02-05)
KrutayaBabka  : refactoring, cleanup and optimization (2026-02-13)

Description
-----------
Common domain data structures for the tracking subsystem.

This module defines lightweight, backend-agnostic representations of detected
and tracked objects (`Object`, `Box`) and a container (`TrackedObjects`) that
adapts raw Ultralytics `Results` into a normalized internal format suitable for
target selection, geometry calculations, and control logic.
"""


import numpy as np
from dataclasses import dataclass

from src.utils.typing_utils import to_ndarray

from typing import Tuple, Optional, List
from ultralytics.engine.results import Results


# =========================
# Domain Models
# =========================

@dataclass(slots=True)
class Box:
    """
    Axis-aligned bounding box in pixel coordinates.

    Parameters
    ----------
    pt1 : tuple of float
        Top-left corner of the box as (x1, y1).
    pt2 : tuple of float
        Bottom-right corner of the box as (x2, y2).

    Notes
    -----
    This class represents a simple 2D rectangular bounding box and does not
    include any class or confidence information. It is intended to be used as
    a geometric primitive inside higher-level objects.
    """

    pt1: Tuple[float, float]
    pt2: Tuple[float, float]


    def as_int(self) -> "Box":
        """
        Return a copy of the box with integer coordinates.

        Returns
        -------
        Box
            New Box instance with coordinates cast to int.
        """

        return Box(
            pt1=(int(self.pt1[0]), int(self.pt1[1])),
            pt2=(int(self.pt2[0]), int(self.pt2[1]))
        )
    

    def pts_to_int(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Return integer corner coordinates of the box.

        This method converts the floating-point corner coordinates of the box
        into integer pixel coordinates without creating a new Box instance.

        Returns
        -------
        tuple of tuple of int
            A pair of points ((x1, y1), (x2, y2)) representing the top-left and
            bottom-right corners of the box in integer pixel coordinates.
        """

        return (
            (int(self.pt1[0]), int(self.pt1[1])),
            (int(self.pt2[0]), int(self.pt2[1]))
        )


@dataclass(slots=True)
class Object:
    """
    Representation of a single detected or tracked object.

    Parameters
    ----------
    cls_id : int
        Numeric class identifier produced by the model.
    cls_name : str
        Human-readable class name.
    conf : float
        Confidence score of the detection or tracking result.
    id : int or None
        Tracking identifier, if available (None for pure detection).
    box : Box
        Bounding box of the object in pixel coordinates.

    Notes
    -----
    This class is a backend-independent abstraction of a detected or tracked
    object. It contains only semantic and geometric information and is suitable
    for use in target selection, filtering, and control logic.
    """

    cls_id: int
    cls_name: str
    conf: float
    id: Optional[int]
    box: Box


@dataclass(slots=True)
class TrackedObjects:
    """
    Unified container for one inference result (one frame).

    This class adapts Ultralytics `Results` into an internal structured format
    consisting of lightweight `Object` instances.

    Attributes
    ----------
    objects : list[Object]
        Parsed detected/tracked objects.
    orig_shape : tuple[int, int]
        Original frame shape as (height, width).
    raw_data : numpy.ndarray or None
        Raw data from model output (if available).
    """

    objects: List[Object]
    orig_shape: Tuple[int, int]
    raw_data: Optional[np.ndarray]


    # =========================
    # Factory
    # =========================

    @classmethod
    def from_results(cls, results: Results) -> "TrackedObjects":
        """
        Build a TrackedObjects instance from Ultralytics Results.

        Parameters
        ----------
        results : Results
            Output produced by an Ultralytics detection or tracking model.

        Returns
        -------
        TrackedObjects
            Normalized tracking result.
        """
        
        results = results.cpu().numpy()
        orig_shape = results.orig_shape

        if results.boxes is None or len(results.boxes) == 0: 
            return cls(
                objects=[],
                orig_shape=orig_shape,
                raw_data=None
            )

        boxes = results.boxes
        n = len(boxes)

        clss = to_ndarray(arr=boxes.cls).astype(int)
        confs = to_ndarray(boxes.conf)
        xyxys = to_ndarray(boxes.xyxy)

        if boxes.id is None: ids = np.full(shape=n, fill_value=None, dtype=object)
        else: ids = to_ndarray(boxes.id).astype(int)

        names = results.names

        objects: List[Object] = []

        for cls_id, conf, box_xyxy, obj_id in zip(clss, confs, xyxys, ids):
            objects.append(
                Object(
                    cls_id=cls_id,
                    cls_name=names.get(cls_id, "Unknown"),
                    conf=conf,
                    id = obj_id,
                    box=Box(
                        pt1=(box_xyxy[0], box_xyxy[1]),
                        pt2=(box_xyxy[2], box_xyxy[3])
                    )
                )
            )

        raw_data = to_ndarray(arr=boxes.data)
        
        return cls(
            objects=objects,
            orig_shape=orig_shape,
            raw_data=raw_data
        )
