"""
src/targeting/aiming.py

Author
------
KrutayaBabka : initial implementation (2026-02-12)

Description
-----------
Module for computing pan and tilt angles from image coordinates using a 
pinhole camera model. Supports both explicit FOV/AOV with origin or a 
camera intrinsic matrix. Provides range-limited angle computation suitable 
for robotic aiming or turret control.

The resulting angles are returned as `Angles` dataclass instances.
"""


import math

from typing import overload, Optional, Tuple
from cv2.typing import MatLike, Size, Point2f
from .types import Angles


class AimingCalculator:
    """
    Calculates pan and tilt angles to aim at a target pixel in an image.

    Supports initialization either with camera intrinsic matrix or with
    field-of-view (FOV), angle-of-view (AOV), and image origin.

    Parameters
    ----------
    image_size : Size
        Size of the input image/frame as (width, height).
    fov : float, optional
        Horizontal field-of-view in radians. Required if camera_matrix is not provided.
    aov : float, optional
        Vertical angle-of-view in radians. Required if camera_matrix is not provided.
    org : Point2f, optional
        Principal point / origin in pixel coordinates. Required if camera_matrix is not provided.
    camera_matrix : MatLike, optional
        3x3 camera intrinsic matrix. Overrides fov/aov/org if provided.
    pan_range : tuple of float, optional
        Minimum and maximum pan angles (degrees). Default is (0, 360).
    tilt_range : tuple of float, optional
        Minimum and maximum tilt angles (degrees). Default is (0, 360).

    Raises
    ------
    ValueError
        If neither camera_matrix nor (fov, aov, org) are provided.

    Attributes
    ----------
    fx : float
        Focal length in pixels along x-axis.
    fy : float
        Focal length in pixels along y-axis.
    org : Point2f
        Image origin / principal point in pixels.
    image_size : Size
        Current image size.
    pan_range : tuple of float
        Allowed pan angle range in degrees.
    tilt_range : tuple of float
        Allowed tilt angle range in degrees.
    """

    @overload
    def __init__(
        self,
        image_size: Size,
        *,
        fov: float,
        aov: float,
        org: Point2f,
        pan_range: Optional[Tuple[float, float]] = None,
        tilt_range: Optional[Tuple[float, float]] = None
    ) -> None:
        ...


    @overload
    def __init__(
        self,
        image_size: Size,
        *,
        camera_matrix: MatLike,
        pan_range: Optional[Tuple[float, float]] = None,
        tilt_range: Optional[Tuple[float, float]] = None
    ) -> None:
        ...


    def __init__(
        self,
        image_size: Size,
        fov: Optional[float] = None,
        aov: Optional[float] = None,
        org: Optional[Point2f] = None,
        camera_matrix: Optional[MatLike] = None,
        pan_range: Optional[Tuple[float, float]] = None,
        tilt_range: Optional[Tuple[float, float]] = None
    ) -> None:
        self.image_size = image_size
        self.pan_range = pan_range if pan_range is not None else (0, 360)
        self.tilt_range = tilt_range if tilt_range is not None else (0, 360)

        if camera_matrix is not None:
            self.fx = camera_matrix[0][0]
            self.fy = camera_matrix[1][1]
            self.org = (camera_matrix[0][2], camera_matrix[1][2])

        elif fov is not None or aov is not None or org is not None:
            self.org: Point2f = org # type: ignore
            # Compute focal lengths in pixels using pinhole camera model
            self.fx = image_size[0] / (-2 * math.tan(fov/2)) #type: ignore
            self.fy = image_size[1] / (-2 * math.tan(aov/2)) #type: ignore

        else:
            raise ValueError(
                "Invalid arguments: provide either "
                "(fov, aov and org) or (camera_matrix)."
            )
        

    def _pinhole_model(self, target: Point2f) -> Angles:
        """
        Compute relative pan and tilt angles to a target using a pinhole model.

        Parameters
        ----------
        target : Point2f
            Pixel coordinates (x, y) of the target in the image.

        Returns
        -------
        Angles
            Pan and tilt angles in degrees relative to the camera's optical axis.
        """

        pan = math.degrees(math.atan((target[0] - self.org[0]) / self.fx))
        tilt = math.degrees(math.atan((target[1] - self.org[1]) / self.fy))

        return Angles(
            pan=pan,
            tilt=tilt
        )


    def change_img_size(self, image_size: Size) -> None:
        """
        Update the image size used for angle calculations.

        Parameters
        ----------
        image_size : Size
            New image size as (width, height).

        Notes
        -----
        Does not automatically update focal lengths; if FOV/AOV are used,
        recomputation of fx/fy is required externally.
        """

        self.image_size = image_size


    def compute_pan_tilt_angles(self, current: Angles, target: Point2f) -> Angles:
        """
        Compute absolute pan and tilt angles to aim at a target.

        Parameters
        ----------
        current : Angles
            Current pan and tilt angles of the device.
        target : Point2f
            Target pixel coordinates in the image.

        Returns
        -------
        Angles
            New pan and tilt angles, constrained within `pan_range` and `tilt_range`.

        Notes
        -----
        - Uses pinhole camera model to convert pixel offsets to angles.
        - Pan increases to the right; tilt decreases upward.
        - Resulting angles are clamped to the specified ranges.
        """
        
        new_angles = self._pinhole_model(target=target)
        pan = current.pan + new_angles.pan
        tilt = current.tilt - new_angles.tilt

        pan = max(self.pan_range[0], min(self.pan_range[1], pan))
        tilt = max(self.tilt_range[0], min(self.tilt_range[1], tilt))

        return Angles(
            pan=pan,
            tilt=tilt
        )
