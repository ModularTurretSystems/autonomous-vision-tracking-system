# src/vision/stereo/calibration_session.py

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

from .capture import StereoCapture
from .types import RawData, StereoFrame
from src.calibration.chessboard import find_chessboard_corners
from src.utils.image import combine_and_resize_frames


class StereoCalibrationSession:
    """
    Manages an interactive session for collecting calibration frames for stereo.
    """

    def __init__(
        self,
        stereo_capture: StereoCapture,
        pattern_size: Tuple[int, int] = (9, 6),
        combined_resolution: Tuple[int, int] = (1280, 720),
        min_good_frames: int = 15,
        max_good_frames: Optional[int] = None,
    ) -> None:
        self.stereo_capture = stereo_capture
        self.pattern_size = pattern_size
        self.combined_resolution = combined_resolution
        self.min_good_frames = min_good_frames
        self.max_good_frames = max_good_frames

        self.SQUARE_SIZE_MM = 30.0
        self.FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        self.WIN_SIZE = (11, 11)
        self.ZERO_ZONE = (-1, -1)
        self.CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.obj_points: List[np.ndarray] = []
        self.img_points_left: List[np.ndarray] = []
        self.img_points_right: List[np.ndarray] = []

        self.objp_template: np.ndarray | None = None
        self._prepare_object_points()  # ← правильный вызов приватного метода

    def _prepare_object_points(self) -> None:
        """Generate 3D coordinates of chessboard corners."""
        nx, ny = self.pattern_size
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        objp *= self.SQUARE_SIZE_MM
        self.objp_template = objp

    def run(self) -> RawData:
        """Starts an interactive session. Returns dict with collected data."""
        winname = "Stereo Camera Calibration"
        collected = 0

        cv2.namedWindow(winname)  # ← обязательно перед imshow

        print("Управление: 's' — сохранить (если доска найдена), 'q' — выйти")

        while True:
            stereo_frame = self.stereo_capture.capture_frame()
            stereo_frame.flip(flipCode=1)

            ret_l, corners_l = find_chessboard_corners(
                img=stereo_frame.frame_l,
                pattern_size=self.pattern_size,
                flags=self.FLAGS,
                refine=True,
                win_size=self.WIN_SIZE,
                zero_zone=self.ZERO_ZONE,
                criteria=self.CRITERIA
            )

            ret_r, corners_r = find_chessboard_corners(
                img=stereo_frame.frame_r,
                pattern_size=self.pattern_size,
                flags=self.FLAGS,
                refine=True,
                win_size=self.WIN_SIZE,
                zero_zone=self.ZERO_ZONE,
                criteria=self.CRITERIA
            )

            display_frame = stereo_frame.copy()

            cv2.drawChessboardCorners(display_frame.frame_l, self.pattern_size, corners_l, ret_l)
            cv2.drawChessboardCorners(display_frame.frame_r, self.pattern_size, corners_r, ret_r)

            combined = combine_and_resize_frames(
                frame_size=self.combined_resolution,
                frames=display_frame.to_list(),
                horizontal=True
            )

            status_text = f"Saved: {self.stereo_capture.get_number_of_frames} | Good: {collected} / min {self.min_good_frames}"
            color = (0, 255, 0) if ret_l and ret_r else (0, 0, 255)
            if ret_l and ret_r:
                status_text += "GOOD"

            cv2.putText(combined, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            cv2.imshow(winname, combined)

            k = cv2.waitKey(1) & 0xFF

            if k == ord('q'):
                print("Выход по 'q'")
                break

            if k == ord('s') and ret_l and ret_r:
                assert self.objp_template is not None, "objp_template не инициализирован"

                self.stereo_capture.save_frame(stereo_frame)

                self.img_points_left.append(corners_l)
                self.img_points_right.append(corners_r)
                self.obj_points.append(self.objp_template.copy())

                collected += 1
                print(f"Сохранён хороший кадр {collected:02d}")

        cv2.destroyAllWindows()

        return RawData(
            collected=collected,
            enough=collected >= self.min_good_frames,
            obj_points = self.obj_points,
            img_points_left = self.img_points_left,
            img_points_right = self.img_points_right,
            saved_images_count = self.stereo_capture.number_of_frames,
            image_size = self.combined_resolution
        )
  