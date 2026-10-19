# Autonomous Vision Tracking System

An autonomous turret control system that uses computer vision (YOLO) to detect and track targets, computes pan/tilt angles from camera calibration data, and drives an Arduino-based turret via serial.

---

## Setup

### 1. Create virtual environment

```bash
py -3.12 -m venv venv      # Windows
python3.12 -m venv venv    # Mac/Linux
```

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

---

## Quick Start Guide

### Step 1 — Capture calibration images

**Single camera:**
```bash
python -m examples.mono.capture_and_detection_demo
```

**Stereo (two cameras):**
```bash
python -m examples.stereo.capture_and_detection_demo
```

> **Important:** If your chessboard is 8×8 squares, set the pattern size to **7×7** (inner corners).
> Capture at least **15 photos** of the chessboard from different angles.

---

### Step 2 — Run calibration

**Single camera:**
```bash
python -m examples.calibration_storage_npz_demo
```

**Stereo:**
```bash
python -m examples.stereo.stereo_calibration_with_npz_demo
```

After calibration, go to `./data/npz/`, move the result file to `./data/`, and rename it to **`RESULT.npz`**.

---

### Step 3 — Run the turret

Connect the Arduino, then:
```bash
python test_servo.py
```

The turret will start tracking autonomously.

---

## Running examples

All example scripts must be run as Python modules from the project root:

```bash
python -m examples.<script_name>
```

Do **not** run them directly with `python examples/script.py` — the package-based import structure will break.

---

## Project Structure

```
autonomous-vision-tracking-system/
├── data/
│   └── frames/
├── examples/
│   ├── mono/
│   │   ├── capture_and_detection_demo.py
│   │   ├── capture_demo.py
│   │   └── mono_calibration_demo.py
│   ├── stereo/
│   │   ├── capture_and_detection_demo.py
│   │   ├── mono_calibration_with_npz_demo.py
│   │   ├── stereo_calibration_demo.py
│   │   └── stereo_calibration_with_npz_demo.py
│   ├── aiming_with_calibrated_camera_demo.py
│   ├── calibration_storage_npz_demo.py
│   ├── camera_characteristics_demo.py
│   ├── capture_and_tracking_demo.py
│   ├── capture_with_error_handling.py
│   ├── charuco_create_demo.py
│   ├── charuco_detection_demo.py
│   ├── chessboard_corners_detection_demo.py
│   └── extract_corners_from_directory_demo.py
├── firmware/
│   └── arduino/
├── src/
│   ├── actuation/
│   │   ├── arduino.py
│   │   └── turret.py
│   ├── calibration/
│   │   ├── dataset/
│   │   │   └── directory.py
│   │   ├── mono/
│   │   │   ├── calibrator.py
│   │   │   ├── constants.py
│   │   │   └── types.py
│   │   ├── patterns/
│   │   │   ├── charuco.py
│   │   │   ├── chessboard.py
│   │   │   └── types.py
│   │   ├── persistence/
│   │   │   └── npz.py
│   │   ├── reporting/
│   │   │   ├── constants.py
│   │   │   └── table.py
│   │   ├── stereo/
│   │   │   ├── calibrator.py
│   │   │   └── types.py
│   │   ├── collect_images.py
│   │   ├── constants.py
│   │   └── types.py
│   ├── camera/
│   │   ├── camera.py
│   │   ├── constants.py
│   │   └── types.py
│   ├── targeting/
│   │   ├── aiming.py
│   │   ├── constants.py
│   │   ├── selector.py
│   │   └── types.py
│   ├── tracking/
│   │   ├── backends/
│   │   │   ├── base.py
│   │   │   ├── config.py
│   │   │   └── yolo.py
│   │   ├── factories/
│   │   │   └── backend_config.py
│   │   ├── config.py
│   │   ├── constants.py
│   │   ├── tracker.py
│   │   └── types.py
│   ├── ui/
│   │   ├── controllers/
│   │   │   └── turret_controller.py
│   │   ├── services/
│   │   │   └── video_service.py
│   │   ├── widgets/
│   │   │   ├── target_overlay.py
│   │   │   └── video_widget.py
│   │   ├── app.py
│   │   └── main_window.py
│   ├── utils/
│   │   ├── camera_utils.py
│   │   ├── cv2_logging.py
│   │   ├── file_utils.py
│   │   ├── image.py
│   │   ├── path.py
│   │   ├── print_options.py
│   │   ├── types.py
│   │   └── typing_utils.py
│   └── vision/
│       ├── mono/
│       │   └── capture.py
│       └── stereo/
│           ├── calibration_session.py
│           ├── capture.py
│           ├── constants.py
│           ├── depth.py
│           ├── system.py
│           └── types.py
├── .gitignore
├── main.ino
├── main.py
├── README.MD
├── requirements.txt
├── run_ui.py
├── test_servo.py
└── yolo26n.pt
```
