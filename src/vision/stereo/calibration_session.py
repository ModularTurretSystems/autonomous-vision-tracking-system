# src/vision/stereo/calibration_session.py

import cv2
from cv2.typing import MatLike
import numpy as np
from typing import Tuple, List, Optional

from .capture import StereoCapture
from .types import RawData
from src.calibration.patterns.chessboard import ChessboardPattern
from src.utils.image import combine_and_resize_frames

from numpy.typing import NDArray
from numpy import float32


# ========== CONSTATNS for run() ==========
WIN_NAME = "Stereo Camera Calibration"

SAVE_BUTTON = 's'
EXIT_BUTTON = 'q'

GOOD_COLOR = (0, 255, 0)
GOOD_STATUS_TEXT = "GOOD"
BAD_COLOR = (0, 0, 255)
BAD_STATUS_TEXT = "BAD"

ORG = (20, 40)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.9
FONT_THICKNESS = 2

# ========== CONSTATNS for __init__() ==========
COMBINED_RESOLUTION = (1280, 720)
MIN_GOOD_FRAMES = 15
SQUARE_SIZE_MM = 30.0

# ==================================================



class StereoCalibrationSession:
    """
    Manages an interactive session for collecting calibration frames for stereo.
    """

    def __init__(
        self,
        stereo_capture: StereoCapture,
        pattern: ChessboardPattern,
        combined_resolution: Tuple[int, int] = COMBINED_RESOLUTION,
        min_good_frames: int = MIN_GOOD_FRAMES,
        max_good_frames: Optional[int] = None,
    ) -> None:
        self.stereo_capture = stereo_capture
        self.pattern = pattern

        self.combined_resolution = combined_resolution
        self.min_good_frames = min_good_frames
        self.max_good_frames = max_good_frames

        self.SQUARE_SIZE_MM = SQUARE_SIZE_MM

        self.obj_points: List[NDArray[float32]] = []
        self.img_points_left: List[MatLike] = []
        self.img_points_right: List[MatLike] = []

        self.objp_template: NDArray[float32]
        self._prepare_object_points()


    def _prepare_object_points(self) -> None:
        """Generate 3D coordinates of chessboard corners."""
        nx, ny = self.pattern.pattern_size
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        objp *= self.SQUARE_SIZE_MM
        self.objp_template = objp


    def run(self) -> RawData:
        """Starts an interactive session. Returns dict with collected data."""
        winname = WIN_NAME
        collected = 0

        cv2.namedWindow(winname)  # ← обязательно перед imshow

        print(f"Управление: {SAVE_BUTTON} — сохранить (если доска найдена), {EXIT_BUTTON} — выйти")

        while True:
            stereo_frame = self.stereo_capture.capture_frame()

            res_l = self.pattern.detect_corners(img=stereo_frame.frame_l)
            res_r = self.pattern.detect_corners(img=stereo_frame.frame_r)

            display_frame = stereo_frame.copy()

            self.pattern.draw_corners(img=display_frame.frame_l, corners=res_l.corners, patternWasFound=res_l.found)
            self.pattern.draw_corners(img=display_frame.frame_r, corners=res_r.corners, patternWasFound=res_r.found)

            combined = combine_and_resize_frames(
                frame_size=self.combined_resolution,
                frames=display_frame.to_list(),
                horizontal=True
            )

            status_text = f"Saved: {self.stereo_capture.get_number_of_frames} | Good: {collected} / min {self.min_good_frames}"
            if res_l.found and res_r.found:
                color = GOOD_COLOR
                status_text += GOOD_STATUS_TEXT
            else:
                color = BAD_COLOR
                status_text += BAD_STATUS_TEXT

            cv2.putText(img=combined, text=status_text, org=ORG, fontFace=FONT, fontScale=FONT_SCALE, color=color, thickness=FONT_THICKNESS)
            cv2.imshow(winname, combined)

            k = cv2.waitKey(delay=1) & 0xFF

            if k == ord(EXIT_BUTTON):
                print(f"Выход по {EXIT_BUTTON}")
                break

            elif k == ord(SAVE_BUTTON) and res_l.found and res_r.found:
                self.stereo_capture.save_frame(stereo_frame)

                self.img_points_left.append(res_l.corners)
                self.img_points_right.append(res_r.corners)
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
  