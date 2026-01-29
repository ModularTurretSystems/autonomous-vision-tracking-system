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
│  │  ├─ constants.py
│  |  └─ types.py
│  ├─ patterns/
│  │  ├─ charuco.py
│  │  ├─ chessboard.py
│  |  └─ types.py
│  ├─ calibrate.py
│  ├─ collect_images.py   
│  ├─ constants.py  
│  └─ types.py  
├─ camera/
│  ├─ camera.py           
│  ├─ constants.py
│  └─ types.py   
├─ tracking/
│  ├─ tracking.py           
│  ├─ types.py  
│  └─ yolo_tracker.py  
├─ utils/
│  ├─ camera_utils.py
│  ├─ cv2_logging.py
│  ├─ file_utils.py
│  ├─ image.py
│  ├─ path.py
│  └─ types.py
├─ vision/
|  ├─ mono/
│  |  └─ capture.py
|  └─ stereo/
│     ├─ calibration_session.py
│     ├─ capture.py
│     ├─ system.py
|     └─ types.py
├─ examples/
│  ├─ mono/
│  |  ├─ capture.py
│  |  ├─ captureAndDetection.py
|  |  └─ monoCalibration.py
│  ├─ stereo/
|  |  └─ capture.py
│  ├─ charucoCreate.py
│  ├─ extractCornersFromDirectory.py
│  ├─ charucoDetection.py
│  ├─ findChessboardCorners.py  
|  └─ testingOverload.py
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
