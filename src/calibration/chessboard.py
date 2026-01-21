import cv2
from cv2.typing import MatLike, Size, TermCriteria
from .constants import ChessboardFlags

def find_chessboard_corners(
    img: MatLike,
    pattern_size: Size,
    *,
    flags: int = ChessboardFlags.DEFAULT,
    refine: bool = True,
    win_size: Size | None = None,
    zero_zone: Size | None = None,
    criteria: TermCriteria | None = None
) -> tuple[bool, MatLike]:
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    ret, corners = cv2.findChessboardCorners(image=gray, patternSize=pattern_size, flags=flags)

    if ret and refine: 
        if win_size is None or zero_zone is None or criteria is None:
            raise ValueError(
                "win_size, zero_zone, and criteria must be provided when refine=True."
            )
        
        corners = cv2.cornerSubPix(image=gray, corners=corners, winSize=win_size, zeroZone=zero_zone, criteria=criteria)

    return ret, corners
