"""
src/tracking/types.py

Author
------
dolby228 : original implementation (2026-02-05)
KrutayaBabka : refactoring and documentation (2026-02-12)

Description
-----------
This module defines common data structures used across the tracking subsystem.

It provides a unified container for inference results produced by tracking
backends (e.g., YOLO-based detectors and trackers). 
"""


import numpy as np
from dataclasses import dataclass

from typing import Tuple, Optional, Dict
from ultralytics.engine.results import Boxes, Masks, Probs, Keypoints, OBB, Results


@dataclass
class TrackedObject:
    """
    Unified representation of a single tracking inference result.

    This class acts as an adapter between Ultralytics `Results` objects and the
    internal tracking pipeline. It stores all available detection and tracking
    outputs in a structured form.

    Parameters
    ----------
    orig_img : numpy.ndarray
        Original input image on which detection/tracking was performed.
    orig_shape : tuple of int
        Shape of the original image as (height, width).
    boxes : Boxes or None
        Bounding box predictions, if available.
    masks : Masks or None
        Segmentation masks, if available.
    probs : Probs or None
        Classification probabilities, if available.
    keypoints : Keypoints or None
        Pose keypoints, if available.
    obb : OBB or None
        Oriented bounding boxes, if available.
    speed : dict of str to float or None
        Timing information for different pipeline stages
        (e.g., preprocessing, inference, postprocessing).
    names : dict of int to str
        Mapping from class indices to class names.
    path : str
        Path to the source image or video frame.
    save_dir : str or None
        Directory where outputs are saved, if saving is enabled.
    """

    orig_img:  np.ndarray
    orig_shape: Tuple[int, int]
    boxes:       Optional[Boxes]
    masks:       Optional[Masks]
    probs:       Optional[Probs]
    keypoints:   Optional[Keypoints]
    obb:         Optional[OBB]
    speed:       Dict[str, Optional[float]]
    names:       Dict[int, str]
    path:        str
    save_dir:    Optional[str]


    @classmethod
    def from_results(cls, results: Results) -> "TrackedObject":
        """
        Create a TrackedObject instance from an Ultralytics Results object.

        This method extracts all relevant fields from the backend-specific
        `Results` object and converts them into a unified `TrackedObject`
        representation used by the tracking subsystem.

        Parameters
        ----------
        results : Results
            Ultralytics inference result produced by a detection or tracking model.

        Returns
        -------
        TrackedObject
            Instance populated with data from the given Results object.
        """

        return cls(
            orig_img=results.orig_img,
            orig_shape=results.orig_shape,
            boxes=results.boxes,
            masks=results.masks,
            probs=results.probs, #type: ignore
            keypoints=results.keypoints,
            obb=results.obb,
            speed=results.speed, #type: ignore
            names=results.names,
            path=results.path,
            save_dir=results.save_dir
        )
