from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import math

from src.tracking.types import AimError


def error_from_dz(
    pt: Tuple[int, int], dz_xyxy: Tuple[int, int, int, int]
) -> Tuple[int, int]:
    x, y = pt
    x1, y1, x2, y2 = dz_xyxy

    if x < x1:
        ex = x - x1
    elif x > x2:
        ex = x - x2
    else:
        ex = 0

    if y < y1:
        ey = y - y1
    elif y > y2:
        ey = y - y2
    else:
        ey = 0

    return int(ex), int(ey)


def normalize_err(
    ex_px: int, ey_px: int, dz_xyxy: Tuple[int, int, int, int]
) -> Tuple[float, float]:
    x1, y1, x2, y2 = dz_xyxy
    hx = max(1, (x2 - x1) // 2)
    hy = max(1, (y2 - y1) // 2)

    ex_norm = float(np.clip(ex_px / hx, -1.0, 1.0))
    ey_norm = float(np.clip(ey_px / hy, -1.0, 1.0))

    return ex_norm, ey_norm


def err_px_to_angle(
    ex_px: int, ey_px: int, fx: float, fy: float
) -> Tuple[float, float]:
    return math.atan(ex_px / fx), math.atan(ey_px / fy)


def compute_aim_error(
    pt: Tuple[int, int],
    dz_xyxy: Tuple[int, int, int, int],
    fx: Optional[float] = None,
    fy: Optional[float] = None,
) -> AimError:
    ex_px, ey_px = error_from_dz(pt=pt, dz_xyxy=dz_xyxy)
    ex_norm, ey_norm = normalize_err(ex_px=ex_px, ey_px=ey_px, dz_xyxy=dz_xyxy)

    ex_ang = ey_ang = None
    if fx is not None and fy is not None:
        ex_ang, ey_ang = err_px_to_angle(ex_px=ex_px, ey_px=ey_px, fx=fx, fy=fy)

    return AimError(
        ex_px=ex_px,
        ey_px=ey_px,
        ex_norm=ex_norm,
        ey_norm=ey_norm,
        ex_ang=ex_ang,
        ey_ang=ey_ang,
    )
