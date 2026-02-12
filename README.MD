## Setup

### 1. Create virtual environment

```bash
py -3.12 -m venv venv      # Windows
python3.12 -m venv venv    # Mac/Linux
````

### 2. Activate virtual environment

```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
src/
├─ calibration/
│  ├─ dataset/
│  |  └─ directory.py
│  ├─ mono/
│  │  ├─ calibrator.py
│  │  ├─ calibrator.pyi
│  │  ├─ constants.py
│  |  └─ types.py
│  ├─ patterns/
│  │  ├─ charuco.py
│  │  ├─ chessboard.py
│  |  └─ types.py
│  ├─ persistence/
│  |  └─ npz.py
│  ├─ reporting/
│  │  ├─ constants.py
│  |  └─ table.py
│  ├─ stereo/
│  │  ├─ calibrator.py
│  |  └─ types.py
│  ├─ collect_images.py   
│  ├─ constants.py  
│  └─ types.py  
├─ camera/
│  ├─ camera.py           
│  ├─ constants.py
│  └─ types.py   
├─ tracking/
|  ├─ backends/
│  |  ├─ base.py           
│  |  ├─ config.py           
│  |  └─ yolo.py  
|  ├─ factories/
│  |  └─ backend_config.py
│  ├─ config.py           
│  ├─ constants.py           
│  ├─ tracker.py           
│  └─ types.py  
├─ utils/
│  ├─ camera_utils.py
│  ├─ cv2_logging.py
│  ├─ file_utils.py
│  ├─ image.py
│  ├─ path.py
│  ├─ print_options.py
│  └─ types.py
├─ vision/
|  ├─ mono/
│  |  └─ capture.py
|  └─ stereo/
│     ├─ calibration_session.py
│     ├─ capture.py
│     ├─ constants.py
│     ├─ system.py
|     └─ types.py
├─ examples/
│  ├─ mono/
│  |  ├─ capture_demo.py
│  |  ├─ capture_and_detection_demo.py
|  |  └─ mono_calibration_demo.py
│  ├─ stereo/
│  |  ├─ capture_and_detection_demo.py
│  |  ├─ mono_calibration_with_npz_demo.py
│  |  ├─ stereo_calibration_demo.py
|  |  └─ stereo_calibration_with_npz_demo.py
│  ├─ calibration_storage_npz_demo.py
│  ├─ capture_with_error_handling.py
│  ├─ charuco_create_demo.py
│  ├─ charuco_detection_demo.py
│  ├─ chessboard_corners_detection_demo.py  
│  └─ extract_corners_from_directory_demo.py  
├─ .gitignore
├─ README.md
├─ requirements.txt
└─ yolo26n.pt
```

## Running examples

All example scripts are located in the `examples/` directory and should be executed as Python modules.

From the project root, activate your virtual environment and run:

```bash
python -m examples.findChessboardCorners
```

⚠️ **Important:**
Do not run example files directly (e.g. `python examples/findChessboardCorners.py`), because the project uses a package-based structure.
Running them as modules ensures that all internal imports work correctly.
