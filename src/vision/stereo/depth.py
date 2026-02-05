from typing import Literal, Optional, Tuple
from src.tracking.disparity import match_disparity_sad_1d, match_disparity_sgbm_strip
import numpy as np
import cv2


class DepthEstimator:
    def __init__(
        self,
        Q: np.ndarray,
        P1: np.ndarray,
        method: Literal["sad", "sgbm"] = "sad",
        depth_every_n: int = 3,
        z_min: float = 500.0,
        z_max: float = 6000.0,
        max_disp: int = 64,
        ema_alpha: float = 0.35,
        max_step: float = 100,
        sad_win: int = 7
    ) -> None:
        self.method = method
        self.depth_every_n = depth_every_n
        self.z_min = z_min
        self.z_max = z_max
        self.max_disp = max_disp
        self.ema_alpha = ema_alpha
        self.max_step = max_step
        self.sad_win = sad_win

        self.f = float(Q[2,3])
        self.B = float(-1/Q[3, 2])

        self.frame_id = 0
        self.last_xR: Optional[int] = None
        self.z_prev: Optional[float] = None

        self.sgbm = None
        if self.method == "sgbm":
            self.sgbm = cv2.StereoSGBM.create(
                minDisparity=0,
                numDisparities=self.max_disp,
                blockSize=5,
                P1=8 * 1 * 5 * 5,
                P2=32 * 1 * 5 * 5,
                uniquenessRatio=10,
                speckleWindowSize=80,
                speckleRange=16,
                disp12MaxDiff=1,
            )

    def update(
        self,
        rect_l_gray: np.ndarray,
        rect_r_gray: np.ndarray,
        pt: Tuple[int, int]
    ) -> Optional[float]:
        self.frame_id += 1
        xL, y = pt

        if self.frame_id % self.depth_every_n != 0:
            return self.z_prev

        if not (0 <= y < rect_l_gray.shape[0]):
            return self.z_prev

        d: Optional[float] = None

        if self.method == "sad":
            row_l = rect_l_gray[y]
            row_r = rect_r_gray[y]
            d = match_disparity_sad_1d(row_l, row_r, xL, max_disp=64, win=7)
        else:
            if self.sgbm is not None:    
                d = match_disparity_sgbm_strip(self.sgbm, rect_l_gray, rect_r_gray, xL, y)

        if d is None or d <= 0:
            return self.z_prev

        xR = round(xL - d)
        self.last_xR = xR

        z_raw = (self.B * self.f) / d
        if not (self.z_min <= z_raw <= self.z_max):
            return self.z_prev

        self.z_prev = self._filter(self.z_prev, z_raw)
        return self.z_prev

    def _filter(
        self,
        z_prev: Optional[float],
        z_raw: float
    ) -> float:
        if z_prev is None: 
            return z_raw

        delta = z_raw - z_prev
        if abs(delta) > self.max_step:
            return z_prev + np.sign(delta) * self.max_step

        return self.ema_alpha * z_raw + (1 - self.ema_alpha) * z_prev
   
    def reset(self) -> None:
        self.z_prev = None
        self.last_xR = None
