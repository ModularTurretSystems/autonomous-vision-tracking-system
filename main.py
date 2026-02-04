from src.calibration import collect_images
from src.calibration.calibrate import save_calibration
from src.tracking.tracking import tracking


def main():
    # save_calibration()
    tracking() 
    


if __name__ == "__main__":
    main()