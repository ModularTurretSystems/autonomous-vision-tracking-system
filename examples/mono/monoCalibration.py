import cv2
from src.calibration.dataset.directory import DirectoryCalibrationDataset
from src.calibration.mono.calibrator import MonoCalibrator
from src.calibration.mono.constants import FixedPointMode
from src.calibration.patterns.chessboard import ChessboardPattern


PATTERN_SIZE = (9, 6)
FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
REFINE = True
WIN_SIZE = (11, 11)
ZERO_ZONE = (-1, -1)
CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.01
)
SQUARE_SIZE = 30.0 # mm

DATA_DIR = "data/frames"

IMAGE_SIZE = (1280, 720)
CALIBRATOR_FLAGS = cv2.CALIB_RATIONAL_MODEL

pattern = ChessboardPattern(pattern_size=PATTERN_SIZE, flags=FLAGS, refine=REFINE, win_size=WIN_SIZE, zero_zone=ZERO_ZONE, criteria=CRITERIA, square_size=SQUARE_SIZE)

data_set = DirectoryCalibrationDataset(data_dir=DATA_DIR, pattern=pattern)
features = data_set.extract_features()

calibrator = MonoCalibrator(pattern=pattern, image_size=IMAGE_SIZE, image_points=features, flags=CALIBRATOR_FLAGS, i_fixed_point=FixedPointMode.TOP_RIGHT)
res = calibrator.calibrate(flags=None)

print(res.rms)
print(res.dist_coeffs)
