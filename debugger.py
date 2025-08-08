"""
StreeRaksha Debug Utility Module
Provides debug utilities for the StreeRaksha system.
"""

import cv2
import numpy as np
import time
import os


class Debugger:
    def __init__(self, logger=None):
        """Initialize the debugger"""
        self.logger = logger
        self.debug_overlay = True
        self.debug_dir = "debug"
        self.frame_history = []
        self.max_frames = 10  # Maximum frames to keep in history

        # Create debug directory
        os.makedirs(self.debug_dir, exist_ok=True)

        # Performance monitoring
        self.fps_history = []
        self.last_time = time.time()

    def log_fps(self):
        """Calculate and log FPS"""
        current_time = time.time()
        fps = 1.0 / \
            (current_time - self.last_time) if (current_time - self.last_time) > 0 else 0
        self.last_time = current_time

        # Keep a history of FPS values
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)

        # Calculate average FPS
        avg_fps = sum(self.fps_history) / len(self.fps_history)

        if self.logger:
            self.logger.debug(f"FPS: {avg_fps:.2f}")

        return avg_fps

    def save_debug_frame(self, frame, suffix="debug"):
        """Save a debug frame"""
        filename = os.path.join(
            self.debug_dir, f"{suffix}_{time.time():.2f}.jpg")
        cv2.imwrite(filename, frame)
        if self.logger:
            self.logger.debug(f"Saved debug frame to {filename}")

    def add_debug_overlay(self, frame, info_dict, position="top-left"):
        """
        Add debug overlay to frame with positioning options

        Parameters:
            frame: The frame to overlay
            info_dict: Dictionary of information to display
            position: Position of overlay - "top-left", "top-right", "bottom-left", "bottom-right"
        """
        if not self.debug_overlay:
            return frame

        # For ultra-low latency, use the same frame without copying
        debug_frame = frame

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Determine starting position and direction based on position parameter
        if position == "top-left":
            x_start, y_start = 10, 30
            x_dir, y_dir = 1, 1  # Right and down
        elif position == "top-right":
            x_start, y_start = width - 200, 30
            x_dir, y_dir = -1, 1  # Left and down
        elif position == "bottom-left":
            x_start, y_start = 10, height - 50
            x_dir, y_dir = 1, -1  # Right and up
        elif position == "bottom-right":
            x_start, y_start = width - 200, height - 50
            x_dir, y_dir = -1, -1  # Left and up
        else:
            # Default to top-left
            x_start, y_start = 10, 30
            x_dir, y_dir = 1, 1

        # Add FPS
        fps = self.log_fps()

        # For minimal overlay, just add critical info in the corner
        is_minimal = len(info_dict) <= 2

        if is_minimal:
            # For minimal display, use larger text and bright colors for visibility
            for i, (key, value) in enumerate(info_dict.items()):
                # Draw with better contrast - black outline with bright color
                text = f"{key}: {value}"
                y = y_start + (i * 30 * y_dir)

                # Highlight latency value with color based on performance
                if "Latency" in key:
                    try:
                        latency_value = float(value.split("ms")[0])
                        if latency_value < 300:
                            color = (0, 255, 0)  # Green for good
                        elif latency_value < 500:
                            color = (0, 255, 255)  # Yellow for acceptable
                        else:
                            color = (0, 0, 255)  # Red for poor
                    except:
                        color = (0, 255, 0)  # Default to green
                else:
                    color = (0, 255, 0)  # Green for other stats

                # Draw outline for better visibility
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    cv2.putText(debug_frame, text,
                                (x_start + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 0), 2, cv2.LINE_AA)

                # Draw text
                cv2.putText(debug_frame, text,
                            (x_start, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2, cv2.LINE_AA)
        else:
            # Regular overlay with all debug information
            y_offset = y_start
            for key, value in info_dict.items():
                cv2.putText(debug_frame, f"{key}: {value}",
                            (x_start, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 1, cv2.LINE_AA)
                y_offset += (30 * y_dir)

        return debug_frame

    def add_frame_to_history(self, frame):
        """Add frame to history"""
        self.frame_history.append(frame.copy())
        if len(self.frame_history) > self.max_frames:
            self.frame_history.pop(0)

    def save_frame_history(self):
        """Save all frames in history"""
        for i, frame in enumerate(self.frame_history):
            self.save_debug_frame(frame, f"history_{i}")

    def draw_track_history(self, frame, persons):
        """Draw track history lines"""
        # Implement track visualization
        return frame
