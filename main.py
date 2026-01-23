from src.calibration import collect_images, calibrate

def main():
    collect_images.collect_images()
    calibrate.out()
    

if __name__ == "__main__":
    main()