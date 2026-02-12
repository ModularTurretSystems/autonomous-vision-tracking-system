"""
src/utils/camera_utils.py

Author
------
dolby228 : original implementation (2026-01-29)
KrutayaBabka : refactoring and documentation (2026-02-12)

Description
-----------
Utilities for computing intrinsic camera parameters and derived
characteristics from a camera matrix. Provides data structures for
encapsulating field-of-view, focal length, principal point, and
aspect ratio.
"""


import cv2
from dataclasses import dataclass

from cv2.typing import MatLike, Size, Point2d
from typing import Optional


@dataclass
class CameraCharacteristics:
    """
    Encapsulates intrinsic camera characteristics derived from
    the camera matrix.

    Parameters
    ----------
    fovx : float
        Horizontal field of view in degrees.
    fovy : float
        Vertical field of view in degrees.
    focal_length : float
        Focal length in pixels.
    principal_point : Point2d
        Coordinates of the principal point (cx, cy) in pixels.
    aspect_ratio : float
        Aspect ratio of the pixels (width / height).
    """

    fovx: float
    fovy: float
    focal_length: float
    principal_point: Point2d
    aspect_ratio: float


def compute_camera_parameters(
    camera_matrix: MatLike,
    image_size: Size,
    *,
    aperture_width: Optional[float] = None,
    aperture_height: Optional[float] = None
) -> CameraCharacteristics:
    """
    Compute intrinsic camera parameters from a camera matrix.

    This function wraps OpenCV's `calibrationMatrixValues` to extract
    the horizontal and vertical field of view, focal length, principal
    point, and pixel aspect ratio from a given camera matrix. Optionally,
    the physical aperture size of the sensor can be provided.

    Parameters
    ----------
    camera_matrix : MatLike
        3x3 intrinsic camera matrix obtained from calibration.
    image_size : Size
        Image dimensions as (width, height) in pixels.
    aperture_width : float, optional
        Physical width of the camera sensor in mm. Required for accurate FOV
        calculation if known. Defaults to None.
    aperture_height : float, optional
        Physical height of the camera sensor in mm. Required for accurate FOV
        calculation if known. Defaults to None.

    Returns
    -------
    CameraCharacteristics
        A dataclass containing the following fields:
        - `fovx`: horizontal field of view in degrees
        - `fovy`: vertical field of view in degrees
        - `focal_length`: focal length in pixels
        - `principal_point`: (cx, cy) coordinates of the principal point in pixels
        - `aspect_ratio`: pixel aspect ratio (width / height)

    Notes
    -----
    - If aperture dimensions are not provided, the FOV will be computed
      assuming default sensor size based on `image_size`.
    """

    fovx, fovy, focal_length, principal_point, aspect_ratio = cv2.calibrationMatrixValues(
        camera_matrix,
        image_size,
        aperture_width,
        aperture_height
    )

    return CameraCharacteristics(
        fovx=fovx,
        fovy=fovy,
        focal_length=focal_length,
        principal_point=principal_point,
        aspect_ratio=aspect_ratio
    )
