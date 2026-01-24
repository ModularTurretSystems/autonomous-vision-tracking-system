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
│  ├─ patterns/
│  │  ├─ charuco.py
│  |  └─ chessboard.py        
│  ├─ calibrate.py
│  ├─ collect_images.py   
│  ├─ constants.py  
│  └─ types.py  
├─ camera/
│  ├─ camera.py           
│  ├─ constants.py
│  └─ types.py        
├─ utils/
│  ├─ camera_utils.py
│  ├─ cv2_logging.py
│  ├─ image.py
│  └─ types.py
├─ vision/
|  └─ stereo/
│     ├─ calibration_session.py
│     ├─ capture.py
│     ├─ system.py
|     └─ types.py
├─ examples/
│  ├─ capture.py
│  ├─ charucoCreate.py
│  ├─ charucoDetection.py
│  ├─ findChessboardCorners.py  
|  └─ testingOverload.py
├─ .gitignore
├─ requirements.txt
└─ README.md
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
