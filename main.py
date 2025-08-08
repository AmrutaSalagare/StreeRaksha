"""
StreeRaksha - Main Application Entry Point

This is the main entry point for the StreeRaksha application.
The application monitors camera feed to detect potential safety concerns for women.

Created by: Stree Raksha Team
"""

import cv2
import time
import warnings
import os
from datetime import datetime

# Import StreeRaksha modules
from detector import StreeRakshaDetector
from gender_detector import GenderDetector
from tracker import PersonTracker
from pose_analyzer import PoseAnalyzer
from alert_system import AlertSystem
from visualizer import Visualizer
from logger import Logger
from debugger import Debugger
from thread_manager import ThreadManager
from firebase_manager import FirebaseManager

# Suppress warnings
warnings.filterwarnings("ignore")


def check_dependencies():
    """Check and report on required dependencies"""
    dependencies = {
        "opencv": True,
        "numpy": True,
        "ultralytics": True,
        "mediapipe": True,
        "tensorflow": True,
        "transformers": True,
    }

    # Check dependencies
    try:
        import numpy
    except ImportError:
        dependencies["numpy"] = False

    try:
        import ultralytics
    except ImportError:
        dependencies["ultralytics"] = False

    try:
        import mediapipe
    except ImportError:
        dependencies["mediapipe"] = False

    try:
        import tensorflow
    except ImportError:
        dependencies["tensorflow"] = False

    try:
        from transformers import AutoModelForImageClassification
    except ImportError:
        dependencies["transformers"] = False

    print("\n=== Dependency Check ===")
    for dep, status in dependencies.items():
        status_msg = "✓ Installed" if status else "✗ Missing"
        print(f"{dep}: {status_msg}")

    if not dependencies["tensorflow"]:
        print("\n====== TensorFlow Installation ======")
        print("TensorFlow is missing but not required for core functionality.")
        print("If you want to install it (for Python 3.10 or 3.11), run:")
        print("pip install tensorflow==2.12.0")
        print("Note: TensorFlow doesn't officially support Python 3.12 yet.")

    if not dependencies["transformers"]:
        print("\n====== Hugging Face Installation ======")
        print("Hugging Face transformers is missing. For better gender detection, install:")
        print("pip install transformers")

    return all(dependencies.values())


def main():
    """Main function to run the StreeRaksha application"""
    print("Starting StreeRaksha Safety Monitoring System...")

    # Initialize logger
    logger = Logger()
    logger.info("Starting StreeRaksha application")

    # Initialize debugger
    debugger = Debugger(logger)

    # Initialize components
    logger.info("Initializing components...")

    # Initialize gender detector
    gender_detector = GenderDetector()

    # Initialize pose analyzer
    pose_analyzer = PoseAnalyzer()

    # Initialize person tracker
    tracker = PersonTracker()

    # Initialize Firebase manager (if config exists)
    firebase_config_path = os.path.join(
        os.path.dirname(__file__), 'firebase_config.json')
    firebase_manager = FirebaseManager(
        logger, config_path=firebase_config_path)

    # Start Firebase upload thread if initialized
    if firebase_manager.initialized:
        firebase_manager.start_upload_thread()

    # Initialize alert system with Firebase manager
    alert_system = AlertSystem(firebase_manager)

    # Initialize visualizer
    visualizer = Visualizer()

    # Add pose analyzer reference to visualizer for multi-threading support
    visualizer.pose_analyzer = pose_analyzer

    try:
        # Initialize YOLO detector
        from ultralytics import YOLO
        logger.info("Loading YOLO model")
        model = YOLO("yolov8n.pt")
        logger.info("YOLO model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        print("Failed to load YOLO model. Please make sure you have ultralytics installed.")
        print("Install with: pip install ultralytics")
        return

    # Initialize thread manager
    thread_manager = ThreadManager(
        model=model,
        tracker=tracker,
        gender_detector=gender_detector,
        pose_analyzer=pose_analyzer,
        alert_system=alert_system,
        visualizer=visualizer,
        debugger=debugger,
        logger=logger
    )

    # Connect to camera
    logger.info("Connecting to camera")
    try:
        cap = connect_camera(logger)
    except Exception as e:
        logger.error(f"Failed to connect to camera: {e}")
        print("Failed to connect to camera. Please check connections and permissions.")
        return

    try:
        # Start thread manager with all threads
        logger.info("Starting multi-threaded processing")
        thread_manager.start(cap)

        # Wait for the stop event to be set
        while not thread_manager.stop_event.is_set():
            time.sleep(0.1)  # Sleep to reduce CPU usage

            # Handle camera reconnection if needed
            if not cap.isOpened():
                logger.warning("Camera disconnected, attempting to reconnect")
                cap.release()
                try:
                    cap = connect_camera(logger)
                    # Update capture thread with new camera
                    thread_manager.stop()
                    thread_manager.start(cap)
                except Exception as e:
                    logger.error(f"Failed to reconnect to camera: {e}")
                    print("Failed to reconnect to camera. Exiting.")
                    break

    except KeyboardInterrupt:
        logger.info("User interrupted program")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        print(f"Error: {e}")
    finally:
        # Clean up
        logger.info("Shutting down StreeRaksha")
        thread_manager.stop()

        # Stop Firebase manager if initialized
        if firebase_manager.initialized:
            firebase_manager.stop()

        cap.release()
        cv2.destroyAllWindows()


def connect_camera(logger):
    """Connect to camera with fallback options"""
    # Try multiple camera indices
    for camera_idx in [0, 1, 2]:
        try:
            print(f"Trying camera index {camera_idx}...")
            cap = cv2.VideoCapture(camera_idx)
            if cap.isOpened():
                print(f"Successfully connected to camera {camera_idx}")
                logger.info(f"Connected to camera index {camera_idx}")
                return cap
        except Exception as e:
            print(f"Failed to connect to camera {camera_idx}: {e}")

    # Fallback to DirectShow backend (Windows)
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            print("Successfully connected using DirectShow")
            logger.info("Connected to camera using DirectShow")
            return cap
    except Exception as e:
        print(f"Failed to connect using DirectShow: {e}")
        logger.error(f"DirectShow camera connection failed: {e}")

    # If all attempts fail
    logger.error("Failed to connect to any camera")
    raise RuntimeError(
        "Could not connect to any camera. Please check connections and permissions.")


if __name__ == "__main__":
    check_dependencies()
    main()
