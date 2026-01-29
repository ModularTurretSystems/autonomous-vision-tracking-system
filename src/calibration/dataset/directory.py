from pathlib import Path
from natsort import natsorted, ns
from cv2 import imread

from src.calibration.patterns.chessboard import ChessboardPattern
from src.utils.path import ensure_directory
from src.utils.file_utils import normalize_extension

from cv2.typing import MatLike
from src.calibration.constants import ImageExtensions


class DirectoryCalibrationDataset:
    def __init__(
        self,
        data_dir: Path | str,
        pattern: ChessboardPattern,
        extentions: str | tuple[str, ...] = ImageExtensions.ALL,
    ) -> None:
        self.data_dir = ensure_directory(path=data_dir)
        self.pattern = pattern

        self.extentions = normalize_extension(extensions=extentions, allowed=ImageExtensions.ALL)


    def extract_features(self) -> list[MatLike]:
        data_paths: list[Path] = []
        for ext in self.extentions: data_paths.extend(self.data_dir.glob(pattern=f"*{ext}", case_sensitive=False)) #type: ignore

        data_paths = natsorted(data_paths, key=lambda p: p.name, alg=ns.PATH | ns.IGNORECASE)
        
        features: list[MatLike] = []

        for path in data_paths:
            img: MatLike = imread(filename=str(path)) #type: ignore
            if img is None: continue # Implement logging here #type: ignore

            res = self.pattern.detect_corners(img=img)
            if not res.found: continue # Implement logging here

            features.append(res.corners)

        return features
    

    def set_data_dir(self, data_dir: Path | str) -> None:
        self.data_dir = ensure_directory(path=data_dir)
    