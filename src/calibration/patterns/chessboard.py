import cv2
import numpy as np
from numpy import float32, zeros, mgrid

from ..constants import ChessboardFlags

from cv2.typing import MatLike, Size, TermCriteria
from typing import Sequence
from .types import ChessBoardDetectionResult


class ChessboardPattern():
    def __init__(
        self,
        pattern_size: Size,
        *,
        flags: int = ChessboardFlags.DEFAULT,
        refine: bool = False,
        win_size: Size | None = None,
        zero_zone: Size | None = None,
        criteria: TermCriteria | None = None,
        square_size: float | None = None
    ) -> None:
        self.pattern_size = pattern_size
        self.flags = flags
        self.refine = refine
        self.win_size = win_size
        self.zero_zone = zero_zone
        self.criteria = criteria

        if square_size is None: self.square_size = None
        else: self.set_square_size(square_size=square_size)

        if self.refine and not all([win_size, zero_zone, criteria]):
            raise ValueError(
                "win_size, zero_zone, and criteria must be provide when refine=True"
            )
            

    def detect_corners(self, img: MatLike) -> ChessBoardDetectionResult:
        gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        
        found = cv2.checkChessboard(img=gray, size=self.pattern_size)
        
        corners = None
        if found: found, corners = cv2.findChessboardCorners(image=gray, patternSize=self.pattern_size, flags=self.flags)

        if found and self.refine:
            corners = cv2.cornerSubPix(image=gray, corners=corners, winSize=self.win_size, zeroZone=self.zero_zone, criteria=self.criteria) #type: ignore

        return ChessBoardDetectionResult(found=found, corners=corners) #type: ignore
    

    def draw_corners(self, img: MatLike, corners: MatLike, patternWasFound: bool) -> MatLike:
        cv2.drawChessboardCorners(image=img, patternSize=self.pattern_size, corners=corners, patternWasFound=patternWasFound)
        return img


    @classmethod
    def is_valid_square_size(cls, square_size: float) -> bool:
        return square_size > 1e-5


    def set_square_size(self, square_size: float) -> None:
        if not self.is_valid_square_size(square_size=square_size):
            raise ValueError(f"Invalid square_size: {square_size}. Must be greater than 1e-5.")

        self.square_size = square_size
    

    def generate_objp(self) -> MatLike:
        if self.square_size is None:
            raise ValueError(
                "square_size is not set. "
                "Please provide it during initialization or via set_square_size() method."
            )
            
        objp = zeros((self.pattern_size[0]*self.pattern_size[1], 3), float32)
        objp[:,:2] = mgrid[0:self.pattern_size[0],0:self.pattern_size[1]].T.reshape(-1,2) * self.square_size

        return objp


    def generate_objps(self, length: int) -> Sequence[MatLike]:
        return list(np.repeat(self.generate_objp()[np.newaxis, :, :], length, axis=0))
