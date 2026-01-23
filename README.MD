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
│  ├─ cv2_logging.py
│  └─ image.py
├─ vision/
|  └─ stereo/
│     ├─ calibration_session.py
│     ├─ capture.py
│     ├─ system.py
|     └─ types.py
├─ .gitignore
├─ requirements.txt
└─ README.md
```
