import numpy as np
from typing import Sequence
import cv2
from cv2.typing import MatLike, Size


def resize_frame(img: MatLike, frame_size: Size) -> MatLike:
    return cv2.resize(src=img, dsize=frame_size, interpolation=cv2.INTER_AREA)


def combine_frames(frames: Sequence[MatLike], horizontal: bool = True) -> MatLike:
    return np.hstack(frames) if horizontal else np.vstack(frames)


def combine_and_resize_frames(frame_size: Size, frames: Sequence[MatLike], horizontal: bool = True) -> MatLike:
    return resize_frame(img=combine_frames(frames=frames, horizontal=horizontal), frame_size=frame_size)
