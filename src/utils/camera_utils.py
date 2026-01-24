import numpy as np

from cv2.typing import MatLike
from typing import Tuple
from .types import AOVParameters


def estimate_AOV(
        matx: MatLike,
        img_size: Tuple[int, int]
) -> AOVParameters:
    
    fx = matx[0, 0] #focal length x
    fy = matx[1, 1] #focal length y
    width, height = img_size
    PI = np.pi

    h_aov = 2 * np.arctan(width / (2 * fx) * (180 / PI))
    v_aov = 2 * np.arctan(height / (2 * fy) * (180 / PI))
    d_aov = 2 * np.arctan(np.sqrt(width ** 2 + height ** 2) / (2 * np.sqrt(fx ** 2 + fy ** 2)) * (180 / PI))

    return AOVParameters(
        horizontal=h_aov,
        vertical=v_aov,
        diagonal=d_aov
    )