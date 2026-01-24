import cv2

from cv2.typing import MatLike, Size, TermCriteria
from ..constants import ChessboardFlags

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
        criteria: TermCriteria | None = None
    ) -> None:
        self.pattern_size = pattern_size
        self.flags = flags
        self.refine = refine
        self.win_size = win_size
        self.zero_zone = zero_zone
        self.criteria = criteria

        if self.refine and not all([win_size, zero_zone, criteria]):
            raise ValueError(
                "win_size, zero_zone, and criteria must be provide when refine=True"
            )
            

    def detect_corners(self, img: MatLike) -> ChessBoardDetectionResult:
        gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        
        found, corners = cv2.findChessboardCorners(image=gray, patternSize=self.pattern_size, flags=self.flags)

        if found and self.refine:
            corners = cv2.cornerSubPix(image=gray, corners=corners, winSize=self.win_size, zeroZone=self.zero_zone, criteria=self.criteria) #type: ignore

        return ChessBoardDetectionResult(found=found, corners=corners) #type: ignore
    

    def draw_corners(self, img: MatLike, corners: MatLike, patternWasFound: bool) -> MatLike:
        cv2.drawChessboardCorners(image=img, patternSize=self.pattern_size, corners=corners, patternWasFound=patternWasFound)
        return img
