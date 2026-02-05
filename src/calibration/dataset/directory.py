from pathlib import Path
from natsort import natsorted, ns
from cv2 import imread

from src.calibration.patterns.chessboard import ChessboardPattern
from src.utils.path import ensure_directory
from src.utils.file_utils import normalize_extension

from cv2.typing import MatLike
from typing import Optional
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

        self.images = None
        self.data_paths = None

          
    def get_data_paths(self) -> list[Path]:
        if self.data_paths is not None: return self.data_paths

        data_paths: list[Path] = []
        for ext in self.extentions:
            data_paths.extend(self.data_dir.glob(pattern=f"*{ext}", case_sensitive=False))
        
        data_paths = natsorted(data_paths, key=lambda p: p.name, alg=ns.PATH | ns.IGNORECASE)

        self.data_paths = data_paths

        return data_paths


    def extract_features(self) -> list[MatLike]:
        features: list[MatLike] = []

        if self.images is None:
            data_paths: list[Path] = self.get_data_paths()
            images: list[MatLike] = []

            for path in data_paths:
                img: Optional[MatLike] = imread(filename=str(path))
                if img is None: continue # Implement logging here

                images.append(img)

                res = self.pattern.detect_corners(img=img)
                if not res.found: continue # Implement logging here

                features.append(res.corners)

            self.images = images
        else:
            for img in self.images:
                res = self.pattern.detect_corners(img=img)
                if not res.found: continue # Implement logging here

                features.append(res.corners)

        return features
    

    def set_data_dir(self, data_dir: Path | str) -> None:
        self.data_dir = ensure_directory(path=data_dir)
        self.data_paths = None
        self.images = None
    

    def load_images(self) -> list[MatLike]:
        data_paths: list[Path] = self.get_data_paths()
        
        images: list[MatLike] = []
        for path in data_paths:
            img: Optional[MatLike] = imread(filename=str(path))
            if img is None: continue # Implement logging here
            
            images.append(img)

        self.images = images

        return images


    def get_images(self) -> list[MatLike]:
        return self.load_images() if self.images is None else self.images
        