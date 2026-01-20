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
├─ __init__.py
├─ calibration/
│  ├─ calibrate.py
│  └─ collect_images.py        
├─ camera/
│  ├─ __init__.py
│  ├─ camera.py           
│  ├─ constants.py
│  └─ types.py        
├─ utils/
│  ├─ __init__.py
│  └─ cv2_logging.py
├─ vision/
│  ├─ __init__.py
|  └─ stereo/
│     ├─ __init__.py
│     ├─ calibration.py
│     ├─ system.py
|     └─ types.py
├─ .gitignore
├─ requirements.txt
└─ README.md
```
