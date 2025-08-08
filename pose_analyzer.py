# pose_analyzer.py

import cv2
import numpy as np
import mediapipe as mp
import time


class PoseAnalyzer:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Use higher complexity model for better accuracy
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Increased from 1 to 2 for better accuracy
            smooth_landmarks=True,
            enable_segmentation=False,  # Don't need segmentation for performance
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Create connection mapping for improved visualization
        self.pose_connections = self.mp_pose.POSE_CONNECTIONS

        # Configuration thresholds
        self.config = {
            'HANDS_UP_THRESHOLD': 0.4,    # Threshold for detecting hands up position
            'LYING_DOWN_THRESHOLD': 0.3,   # Threshold for detecting lying down
            'PROXIMITY_THRESHOLD': 120,    # Pixel distance for proximity alerts
            'KNEE_BEND_THRESHOLD': 120,    # Angle threshold for crouching detection
            'RUNNING_STRIDE_THRESHOLD': 0.2  # Relative threshold for running detection
        }

    def detect_pose(self, frame, person_box):
        """Detect and analyze pose within a person bounding box with improved accuracy"""
        try:
            # Extract person region with padding for better full-body detection
            x1, y1, x2, y2 = person_box

            # Add padding (10% on each side) to include more context around the person
            # This helps MediaPipe detect the pose more accurately
            height, width = frame.shape[:2]
            padding_x = int((x2 - x1) * 0.1)
            padding_y = int((y2 - y1) * 0.1)

            # Apply padding but keep within image boundaries
            x1_pad = max(0, x1 - padding_x)
            y1_pad = max(0, y1 - padding_y)
            x2_pad = min(width, x2 + padding_x)
            y2_pad = min(height, y2 + padding_y)

            # Extract the padded region
            person_img = frame[y1_pad:y2_pad, x1_pad:x2_pad]

            if person_img.size == 0:
                return None, None

            # Convert to RGB for MediaPipe
            rgb_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

            # Process the image
            start_time = time.time()
            results = self.pose_detector.process(rgb_img)
            process_time = time.time() - start_time

            # Debug processing time
            # print(f"Pose detection took {process_time*1000:.2f}ms")

            if not results.pose_landmarks:
                return None, None

            # Normalize landmarks to the original frame coordinates
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                # Convert relative coordinates within the cropped image to absolute coordinates in the frame
                px = int(landmark.x * person_img.shape[1]) + x1_pad
                py = int(landmark.y * person_img.shape[0]) + y1_pad
                visibility = landmark.visibility
                landmarks.append((px, py, visibility))

            # Analyze pose
            pose_analysis = self.analyze_pose(landmarks)

            return landmarks, pose_analysis

        except Exception as e:
            print(f"Error in pose detection: {e}")
            return None, None

    def analyze_pose(self, landmarks):
        """Analyze pose landmarks to detect specific poses"""
        if not landmarks or len(landmarks) < 33:  # MediaPipe provides 33 landmarks
            return {}

        analysis = {
            'hands_up': False,
            'lying_down': False,
            'crouching': False,
            'running': False
        }

        # Extract key landmarks
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]

        # Check hands up position
        if (left_wrist[1] < left_shoulder[1] - self.config['HANDS_UP_THRESHOLD'] * (left_hip[1] - left_shoulder[1]) and
            left_wrist[2] > 0.5) or (right_wrist[1] < right_shoulder[1] - self.config['HANDS_UP_THRESHOLD'] * (right_hip[1] - right_shoulder[1]) and
                                     right_wrist[2] > 0.5):
            analysis['hands_up'] = True

        # Check lying down position
        shoulder_hip_angle = abs(
            (left_shoulder[1] + right_shoulder[1])/2 -
            (left_hip[1] + right_hip[1])/2
        ) / ((left_hip[1] - left_shoulder[1]) if left_hip[1] != left_shoulder[1] else 1)

        if shoulder_hip_angle < self.config['LYING_DOWN_THRESHOLD']:
            analysis['lying_down'] = True

        # Check crouching position
        knee_angle = self.calculate_knee_bend(landmarks)
        if knee_angle and knee_angle < self.config['KNEE_BEND_THRESHOLD']:
            analysis['crouching'] = True

        # Check running motion
        ankle_distance = (
            (left_ankle[0] - right_ankle[0])**2 + (left_ankle[1] - right_ankle[1])**2)**0.5
        ankle_height_diff = abs(left_ankle[1] - right_ankle[1])
        hip_width = abs(right_hip[0] - left_hip[0])

        if (ankle_distance > self.config['RUNNING_STRIDE_THRESHOLD'] * hip_width and
                ankle_height_diff > 20):
            analysis['running'] = True

        return analysis

    def calculate_knee_bend(self, landmarks):
        """Calculate approximate knee bend angle"""
        try:
            # Get hip, knee, ankle landmarks
            left_hip = np.array(landmarks[23][:2])
            left_knee = np.array(landmarks[25][:2])
            left_ankle = np.array(landmarks[27][:2])

            # Calculate vectors
            hip_to_knee = left_knee - left_hip
            ankle_to_knee = left_knee - left_ankle

            # Calculate angle
            dot_product = np.dot(hip_to_knee, ankle_to_knee)
            norm_product = np.linalg.norm(
                hip_to_knee) * np.linalg.norm(ankle_to_knee)

            if norm_product == 0:
                return None

            angle_rad = np.arccos(
                np.clip(dot_product / norm_product, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)

            return angle_deg
        except:
            return None

    def draw_pose(self, frame, landmarks):
        """Draw enhanced pose skeleton and landmarks on the frame using MediaPipe's connections"""
        if not landmarks:
            return frame

        # Convert our landmarks format to MediaPipe format for drawing
        mp_landmarks = self.mp_pose.PoseLandmark

        # Create a temporary overlay for enhanced visualization
        overlay = frame.copy()

        # Dictionary for better color-coding different body parts
        body_part_colors = {
            "torso": (0, 255, 255),     # Yellow
            "left_arm": (0, 165, 255),  # Orange
            "right_arm": (0, 165, 255),  # Orange
            "left_leg": (255, 255, 0),  # Cyan
            "right_leg": (255, 255, 0),  # Cyan
            "face": (255, 0, 255)       # Magenta
        }

        # Map body parts to their MediaPipe connections
        body_parts = {
            "torso": [
                (mp_landmarks.LEFT_SHOULDER, mp_landmarks.RIGHT_SHOULDER),
                (mp_landmarks.RIGHT_SHOULDER, mp_landmarks.RIGHT_HIP),
                (mp_landmarks.RIGHT_HIP, mp_landmarks.LEFT_HIP),
                (mp_landmarks.LEFT_HIP, mp_landmarks.LEFT_SHOULDER)
            ],
            "left_arm": [
                (mp_landmarks.LEFT_SHOULDER, mp_landmarks.LEFT_ELBOW),
                (mp_landmarks.LEFT_ELBOW, mp_landmarks.LEFT_WRIST)
            ],
            "right_arm": [
                (mp_landmarks.RIGHT_SHOULDER, mp_landmarks.RIGHT_ELBOW),
                (mp_landmarks.RIGHT_ELBOW, mp_landmarks.RIGHT_WRIST)
            ],
            "left_leg": [
                (mp_landmarks.LEFT_HIP, mp_landmarks.LEFT_KNEE),
                (mp_landmarks.LEFT_KNEE, mp_landmarks.LEFT_ANKLE),
                (mp_landmarks.LEFT_ANKLE, mp_landmarks.LEFT_FOOT_INDEX)
            ],
            "right_leg": [
                (mp_landmarks.RIGHT_HIP, mp_landmarks.RIGHT_KNEE),
                (mp_landmarks.RIGHT_KNEE, mp_landmarks.RIGHT_ANKLE),
                (mp_landmarks.RIGHT_ANKLE, mp_landmarks.RIGHT_FOOT_INDEX)
            ]
        }

        # Draw skeleton with enhanced thickness and color coding by body part
        for body_part, connections in body_parts.items():
            color = body_part_colors[body_part]
            for connection in connections:
                start_idx = connection[0].value
                end_idx = connection[1].value

                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_landmark = landmarks[start_idx]
                    end_landmark = landmarks[end_idx]

                    # Only draw if both points have reasonable visibility
                    if start_landmark[2] > 0.3 and end_landmark[2] > 0.3:
                        start_point = (
                            int(start_landmark[0]), int(start_landmark[1]))
                        end_point = (
                            int(end_landmark[0]), int(end_landmark[1]))

                        # Thicker lines for better visibility
                        cv2.line(overlay, start_point, end_point, color, 3)

        # Draw landmarks with varying sizes based on visibility
        for landmark in landmarks:
            x, y, visibility = landmark
            if visibility > 0.3:  # Lower threshold to show more points
                # Vary the size based on visibility
                radius = int(4 * visibility) + 1
                # Green for highly visible points, more transparent for less visible
                color = (0, int(255 * visibility), 0)
                cv2.circle(overlay, (int(x), int(y)), radius, color, -1)

        # Blend the overlay with the original frame for a cleaner look
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame
