import cv2
import time

def reset_camera():
    print("Attempting to reset camera resources...")
    
    # Try multiple approaches
    for index in range(5):
        try:
            print(f"Resetting camera index {index}...")
            # Try to open and immediately release
            cap = cv2.VideoCapture(index)
            time.sleep(0.5)
            cap.release()
            time.sleep(0.5)
        except Exception as e:
            print(f"Error with camera {index}: {e}")
    
    print("Camera reset complete")
    print("Wait a few seconds before trying to use the camera again")

if __name__ == "__main__":
    reset_camera()