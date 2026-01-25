import cv2

from src.calibration.dataset.directory import DirectoryCalibrationDataset
from src.calibration.patterns.chessboard import ChessboardPattern


DATA_DIR = "data/frames/"

PATTERN_SIZE = (9, 6)
FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
# FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
REFINE = True
WIN_SIZE = (11, 11)
ZERO_ZONE = (-1, -1)
CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.01
)

pattern = ChessboardPattern(pattern_size=PATTERN_SIZE, flags=FLAGS, refine=REFINE, win_size=WIN_SIZE, zero_zone=ZERO_ZONE, criteria=CRITERIA)

dataset = DirectoryCalibrationDataset(data_dir=DATA_DIR, pattern=pattern)

features = dataset.extract_features()

print(len(features))
