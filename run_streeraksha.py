"""
StreeRaksha Run Script
Helper script to run StreeRaksha with optimized multi-threading settings.
"""

import os
import sys
import argparse
import json
import psutil
import time
from datetime import datetime


def get_system_info():
    """Get basic system information for optimizing thread settings"""
    info = {
        "cpu_count": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "platform": sys.platform
    }
    return info


def create_default_config(system_info):
    """Create default configuration based on system capabilities"""
    config = {
        "version": "1.0.0",
        "last_updated": datetime.now().isoformat(),
        "system": {
            "debug_mode": False,
            "save_evidence": True,
            "evidence_quality": 95,
            "log_level": "INFO"
        },
        "threading": {
            "frame_queue_size": min(30, max(10, system_info["cpu_count"] * 3)),
            "result_queue_size": min(10, max(5, system_info["cpu_count"])),
            "upload_queue_size": 100,
            "capture_thread_priority": "NORMAL",
            "processing_thread_priority": "ABOVE_NORMAL" if system_info["cpu_count"] >= 4 else "NORMAL",
            "rendering_thread_priority": "NORMAL",
            "upload_thread_priority": "BELOW_NORMAL",
            "max_fps": 30 if system_info["memory_gb"] >= 8 else 20
        },
        "detection": {
            "model_confidence": 0.5,
            "person_min_size": 60,
            "nms_threshold": 0.4,
            "track_expiration": 8,
            "max_track_distance": 50
        },
        "camera": {
            "index": 0,
            "width": 640 if system_info["memory_gb"] < 8 else 1280,
            "height": 480 if system_info["memory_gb"] < 8 else 720,
            "fps": 20 if system_info["memory_gb"] < 8 else 30
        }
    }
    return config


def main():
    """Main function to run StreeRaksha with optimized settings"""
    parser = argparse.ArgumentParser(
        description='Run StreeRaksha with optimized multi-threading')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (default: 0)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--config', type=str,
                        help='Path to custom config file')
    parser.add_argument('--max-fps', type=int,
                        help='Maximum frames per second')
    parser.add_argument('--performance', choices=['low', 'medium', 'high', 'realtime', 'ultralow'], default='medium',
                        help='Performance mode: low, medium, high, realtime, or ultralow (minimum latency)')

    args = parser.parse_args()

    # Get system information
    system_info = get_system_info()
    print(f"\n=== System Information ===")
    print(
        f"CPU: {system_info['physical_cores']} physical cores, {system_info['cpu_count']} logical cores")
    print(f"Memory: {system_info['memory_gb']} GB")
    print(f"Platform: {system_info['platform']}")

    # Create config path if it doesn't exist
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')

    # Load existing config or create new one
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            print(f"Loaded custom config from {args.config}")
        except Exception as e:
            print(f"Error loading custom config: {e}")
            config = create_default_config(system_info)
    elif os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Loaded existing config from {config_path}")
        except Exception as e:
            print(f"Error loading existing config: {e}")
            config = create_default_config(system_info)
    else:
        config = create_default_config(system_info)
        print(f"Created new configuration")

    # Override with command line arguments
    if args.camera is not None:
        config["camera"]["index"] = args.camera

    if args.debug:
        config["system"]["debug_mode"] = True
        config["system"]["log_level"] = "DEBUG"

    if args.max_fps:
        config["threading"]["max_fps"] = args.max_fps

    # Apply performance presets - Improved for reduced lag
    if args.performance == 'low':
        config["camera"]["width"] = 640
        config["camera"]["height"] = 480
        config["camera"]["fps"] = 15
        # Process fewer frames than capture
        config["threading"]["max_fps"] = 12
        # Smaller queue to reduce latency
        config["threading"]["frame_queue_size"] = 5
        # Higher confidence to process fewer detections
        config["detection"]["model_confidence"] = 0.6
    elif args.performance == 'medium':
        config["camera"]["width"] = 848
        config["camera"]["height"] = 480
        config["camera"]["fps"] = 20
        # Process fewer frames than capture
        config["threading"]["max_fps"] = 15
        config["threading"]["frame_queue_size"] = 8  # Balanced queue size
    elif args.performance == 'high':
        config["camera"]["width"] = 1280
        config["camera"]["height"] = 720
        config["camera"]["fps"] = 30
        # Even on high, process fewer frames than capture
        config["threading"]["max_fps"] = 20
        # Larger but still limited queue
        config["threading"]["frame_queue_size"] = 12
    elif args.performance == 'realtime':
        # Realtime mode prioritizes low latency over quality
        config["camera"]["width"] = 640
        config["camera"]["height"] = 480
        # Capture at higher rate but process fewer frames
        config["camera"]["fps"] = 30
        config["threading"]["max_fps"] = 15
        # Very small queue to minimize latency
        config["threading"]["frame_queue_size"] = 3
        config["detection"]["model_confidence"] = 0.6
        # Add a note about realtime mode
        print("\n[REALTIME MODE] Prioritizing low latency over quality and accuracy.")
    elif args.performance == 'ultralow':
        # Ultra-low latency mode - absolute minimal processing delay
        # Reduced resolution for faster processing
        config["camera"]["width"] = 416
        config["camera"]["height"] = 320
        # Higher capture rate to get freshest frames
        config["camera"]["fps"] = 60
        config["threading"]["max_fps"] = 60
        # Minimum queue size for direct frame transfer
        config["threading"]["frame_queue_size"] = 1
        config["threading"]["result_queue_size"] = 1
        # Higher confidence threshold for faster detections
        config["detection"]["model_confidence"] = 0.75
        # Skip all non-essential processing
        if "processing" not in config:
            config["processing"] = {}
        config["processing"]["skip_gender_detection"] = True
        config["processing"]["minimal_tracking"] = True
        config["processing"]["skip_pose_estimation"] = True
        config["processing"]["direct_rendering"] = True
        # Enable zero-copy frame transfers
        config["processing"]["zero_copy_mode"] = True
        # Reduce visualization overhead
        config["processing"]["minimal_visualization"] = True
        # Allow early detection termination
        config["processing"]["early_exit_detection"] = True
        # Only process the largest detection
        config["processing"]["max_detections"] = 1
        print("\n[ULTRA-LOW LATENCY MODE] Sacrificing features for sub-0.5s delay.")

    # Save the config for future use
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving configuration: {e}")

    print("\n=== Starting StreeRaksha ===")
    print(f"Camera: {config['camera']['index']}")
    print(
        f"Resolution: {config['camera']['width']}x{config['camera']['height']}")
    print(f"Max FPS: {config['threading']['max_fps']}")
    print(
        f"Debug Mode: {'Enabled' if config['system']['debug_mode'] else 'Disabled'}")
    print("Starting in 3 seconds...")
    time.sleep(3)

    # Import and run main application
    try:
        import main
        main.main()
    except KeyboardInterrupt:
        print("\nStreeRaksha stopped by user")
    except Exception as e:
        print(f"\nError running StreeRaksha: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
