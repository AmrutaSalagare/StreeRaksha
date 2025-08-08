"""
StreeRaksha Visualization Module
Handles UI rendering and visualization.
"""

import cv2
import numpy as np
from datetime import datetime


class Visualizer:
    def __init__(self):
        """Initialize the visualizer"""
        self.pose_analyzer = None  # Will be set by main.py for threading support

    def draw_detection_boxes(self, frame, persons):
        """Draw detection boxes for all persons"""
        for person in persons:
            # Extract info from tracker
            bbox = person['bbox']
            x1, y1, x2, y2 = bbox
            gender = person.get('gender', 'Unknown')
            gender_confidence = person.get('gender_confidence', 0.0)
            track_id = person.get('track_id', '')
            # Set color - Red for female (0,0,255), Blue for male (255,0,0) in BGR
            color = (0, 0, 255) if gender == "Female" else (255, 0, 0)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Add label with gender and confidence
            label = f"{gender}: {gender_confidence:.2f}"
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw track ID if needed
            if track_id:
                id_pos = (x1, y1-30) if gender_confidence > 0 else (x1, y1-10)
                cv2.putText(frame, f"ID:{track_id}", id_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Draw pose labels if any pose detected
            if 'pose' in person:
                pose = person['pose']
                pose_alerts = []
                if pose.get('hands_up', False):
                    pose_alerts.append("Hands Up")
                if pose.get('lying_down', False):
                    pose_alerts.append("Lying Down")
                if pose.get('running', False):
                    pose_alerts.append("Running")
                if pose.get('crouching', False):
                    pose_alerts.append("Crouching")

                if pose_alerts:
                    pose_text = " & ".join(pose_alerts)
                    cv2.putText(frame, pose_text, (x1, y2+15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame

    def draw_stats(self, frame, frame_count, persons, current_time=None, latency=None):
        """Draw statistics overlay with optional latency information"""
        if current_time is None:
            current_time = datetime.now()

        # Calculate statistics
        females = [p for p in persons if p.get('gender') == "Female"]
        males = [p for p in persons if p.get('gender') == "Male"]

        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]

        # Prepare stats text
        stats_text = [
            f"Frame: {frame_count}",
            f"Persons: {len(persons)}",
            f"Female: {len(females)}",
            f"Male: {len(males)}",
            f"Time: {current_time.strftime('%H:%M:%S')}"
        ]

        # Add latency information if available
        if latency is not None:
            latency_ms = latency * 1000  # Convert to milliseconds
            stats_text.append(f"Latency: {latency_ms:.1f}ms")

        # Draw stats at the bottom of the frame
        for i, text in enumerate(stats_text):
            cv2.putText(frame, text,
                        (10, frame_height - 20 - i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def draw_alert(self, frame, alert_message):
        """Draw alert message"""
        if alert_message:
            # Add large alert message at the top
            cv2.putText(frame, alert_message,
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2, cv2.LINE_AA)

        return frame
