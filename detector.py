"""
StreeRaksha Detector Module
Contains the main detector class that processes video frames and coordinates the detection pipeline.
"""

import os
import cv2
import time
import torch
import warnings
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
from pose_analyzer import PoseAnalyzer

# Try importing optional dependencies
try:
    import tensorflow as tf
    has_tensorflow = True
except ImportError:
    has_tensorflow = False
    print("TensorFlow not available. Continuing without TensorFlow functionality.")

try:
    from transformers import AutoFeatureExtractor, AutoModelForImageClassification
    has_transformers = True
except ImportError:
    has_transformers = False
    print("Hugging Face transformers not available. Falling back to simulated gender detection.")

try:
    import mediapipe as mp
    has_mediapipe = True
    print("MediaPipe successfully imported for pose detection")
except ImportError:
    has_mediapipe = False
    print("MediaPipe not available. Install with: pip install mediapipe")


class StreeRakshaDetector:
    def __init__(self):
        # Create directories for evidence
        os.makedirs('evidence', exist_ok=True)

        # Setup logging directory
        os.makedirs('logs', exist_ok=True)
        self.log_filename = f"StreeRaksha_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.log_path = os.path.join('logs', self.log_filename)

        # Configuration for optimization
        self.TRACK_EXPIRATION = 8  # Number of frames before we expire a track
        self.MAX_TRACK_DISTANCE = 50  # Maximum distance for track association
        self.DETECTION_CONFIDENCE = 0.5
        self.MINIMUM_PERSON_SIZE = 60  # Minimum width or height for a valid person detection

        # Unified person trackers
        self.person_trackers = {}
        self.next_track_id = 0

        # Person gender cache
        self.person_gender_cache = {}

        # Last alert time for cooldown
        self.last_alert_time = 0

        # Load models
        self._load_models()

        # Alert configuration
        self.alert_config = {
            'NIGHT_HOURS_START': 18,
            'NIGHT_HOURS_END': 6,
            'MEN_RATIO_THRESHOLD': 1.0,
            'PROXIMITY_THRESHOLD': 150,
            'ALERT_COOLDOWN': 10.0,
            'RISK_THRESHOLD': 60,     # Minimum risk score to trigger alert
            'FRAME_THRESHOLD': 5      # Number of consecutive frames before alert
        }

        # Alert levels
        self.alert_levels = {
            "NOTICE": 1,    # Might be concerning
            "WARNING": 2,   # Potentially troubling
            "ALERT": 3      # Serious concern
        }

        # Initialize pose analyzer
        self.pose_analyzer = PoseAnalyzer()

        # Potential alerts storage
        self.potential_alerts = {}  # Track {alert_type: consecutive_frames}

        # Debug flags
        self.show_risk_scores = True  # Set to True to show risk scores on screen

        # Log initialization
        self.log_message("StreeRaksha initialized")

    def log_message(self, message):
        """Log a message to file with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"

        # Append to log file
        with open(self.log_path, 'a') as log_file:
            log_file.write(log_entry)

    def _load_models(self):
        """Load all required models"""
        print("Loading YOLO model...")
        try:
            # This will download the model if not already present
            self.model = YOLO('yolov8n.pt')
            self.has_yolo = True
            print("Successfully loaded YOLO model")
            self.log_message("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.has_yolo = False
            print("⚠️ YOLO model could not be loaded. Person detection will not work!")
            self.log_message(f"YOLO model load failed: {e}")

        print("Loading face detector...")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.log_message("Face detector loaded")
        print("Loading gender classification model from Hugging Face...")
        self.use_gender_model = False
        if has_transformers:
            try:
                self.gender_feature_extractor = AutoFeatureExtractor.from_pretrained(
                    "rizvandwiki/gender-classification-2")
                self.gender_model = AutoModelForImageClassification.from_pretrained(
                    "rizvandwiki/gender-classification-2")
                self.use_gender_model = True
                print("Successfully loaded gender classification model")
                self.log_message(
                    "Gender classification model loaded successfully")
            except Exception as e:
                print(f"Error loading gender model: {e}")
                print("Falling back to simulated gender classification")
                self.log_message(f"Gender model load failed: {e}")
        else:
            self.use_gender_model = False
            print(
                "Hugging Face transformers not available. Using simulated gender classification")
            self.log_message("Using simulated gender classification")

    def connect_camera(self):
        """Connect to camera with fallback options"""
        # Try multiple camera indices
        for camera_idx in [0, 1, 2]:
            try:
                print(f"Trying camera index {camera_idx}...")
                cap = cv2.VideoCapture(camera_idx)
                if cap.isOpened():
                    print(f"Successfully connected to camera {camera_idx}")
                    self.log_message(f"Connected to camera index {camera_idx}")
                    return cap
            except Exception as e:
                print(f"Failed to connect to camera {camera_idx}: {e}")

        # Fallback to DirectShow backend (Windows)
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if cap.isOpened():
                print("Successfully connected using DirectShow")
                self.log_message("Connected to camera using DirectShow")
                return cap
        except Exception as e:
            print(f"Failed to connect using DirectShow: {e}")
            # If all attempts fail
            self.log_message(f"DirectShow camera connection failed: {e}")
        self.log_message("Failed to connect to any camera")
        raise RuntimeError(
            "Could not connect to any camera. Please check connections and permissions.")

    def predict_gender(self, image):
        """Predict gender using Hugging Face model with stronger male bias correction"""
        if not self.use_gender_model:
            # Fallback to simulated gender if model not available
            # Use the hash of the image data as a deterministic seed
            seed_value = hash(image.tobytes()) % 100
            gender = "Female" if seed_value < 30 else "Male"  # 30% chance female
            confidence = np.random.uniform(0.80, 0.95)
            return gender, confidence

        try:
            # Convert OpenCV BGR image to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # Prepare image for the model
            inputs = self.gender_feature_extractor(
                images=pil_image, return_tensors="pt")

            # Get prediction
            with torch.no_grad():
                outputs = self.gender_model(**inputs)
                logits = outputs.logits

                # Find indices for male and female classes
                male_idx = None
                female_idx = None
                for idx, label in self.gender_model.config.id2label.items():
                    if "female" in label.lower() or "woman" in label.lower() or "f" == label.lower():
                        female_idx = idx
                    else:
                        male_idx = idx

                # Apply a stronger bias toward male classification
                if male_idx is not None and female_idx is not None:
                    bias_correction = 0.25  # Increased bias factor
                    logits[0][male_idx] += bias_correction

                predicted_class_idx = logits.argmax(-1).item()

            # Get confidence
            softmax_values = torch.softmax(logits, dim=1)[0]
            confidence = softmax_values[predicted_class_idx].item()

            # Get gender label
            gender_label = self.gender_model.config.id2label[predicted_class_idx]

            # Format as "Male" or "Female"
            # Higher threshold for female classification based on image quality
            female_threshold = 0.60
            male_threshold = 0.40    # Slightly decreased to favor male classification

            if "female" in gender_label.lower() or "woman" in gender_label.lower() or "f" == gender_label.lower():
                gender = "Female"

                # If female but confidence is low, check the gap with male prediction
                if confidence < female_threshold and male_idx is not None:
                    male_confidence = softmax_values[male_idx].item()

                    # If the difference in confidence is small, classify as male
                    if (confidence - male_confidence) < 0.3:  # Increased from 0.15
                        gender = "Male"
                        confidence = male_confidence
            else:
                gender = "Male"

            # Add small confidence boost for male classification
            if gender == "Male" and confidence < 0.9:
                confidence = min(0.9, confidence * 1.1)

            return gender, confidence

        except Exception as e:
            # On failure, use deterministic fallback
            print(f"Gender prediction error: {e}")
            self.log_message(f"Gender prediction error: {e}")
            seed_value = hash(image.tobytes()) % 100
            gender = "Female" if seed_value < 30 else "Male"  # 30% chance female
            confidence = np.random.uniform(0.80, 0.95)
            return gender, confidence

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate areas of both boxes
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        # Calculate IoU
        iou = intersection_area / \
            float(box1_area + box2_area - intersection_area)
        return max(0.0, min(1.0, iou))  # Ensure value is between 0 and 1

    def center_distance(self, box1, box2):
        """Calculate center distance between boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
        center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)

        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def non_max_suppression(self, boxes, iou_threshold=0.5):
        """Non-maximum suppression to remove overlapping boxes"""
        if not boxes:
            return []

        # Sort boxes by area (largest first)
        sorted_boxes = sorted(boxes, key=lambda x: (
            x[2]-x[0]) * (x[3]-x[1]), reverse=True)

        keep = []
        while sorted_boxes:
            current_box = sorted_boxes.pop(0)
            keep.append(current_box)

            # Filter out boxes with high IoU
            sorted_boxes = [
                box for box in sorted_boxes
                if self.calculate_iou(current_box, box) < iou_threshold
            ]

        return keep

    def calculate_risk_score(self, female, females, males, is_night, persons):
        """Calculate comprehensive risk score based on multiple factors"""
        score = 0

        # Base environmental risk
        if is_night:
            score += 15  # Night time increases baseline risk

        # Male proximity risk
        f_x1, f_y1, f_x2, f_y2 = female['bbox']
        f_center_x = (f_x1 + f_x2) / 2
        f_center_y = (f_y1 + f_y2) / 2

        nearby_males = 0
        close_males = 0

        for male in males:
            m_x1, m_y1, m_x2, m_y2 = male['bbox']
            m_center_x = (m_x1 + m_x2) / 2
            m_center_y = (m_y1 + m_y2) / 2

            distance = np.sqrt(
                (f_center_x - m_center_x)**2 +
                (f_center_y - m_center_y)**2
            )

            if distance < self.alert_config['PROXIMITY_THRESHOLD']:
                nearby_males += 1
                score += 15  # Male in proximity increases risk

            if distance < self.alert_config['PROXIMITY_THRESHOLD'] * 0.5:
                close_males += 1
                score += 15  # Very close male increases risk significantly

        # Surrounding risk
        if nearby_males >= 2:
            score += 20  # Multiple males nearby increases risk more than linearly

        # Behavioral risk - highest weight factors
        if 'pose' in female:
            pose = female.get('pose', {})

            # Check for vulnerable postures
            if pose.get('hands_up', False):
                score += 30  # Possible defensive posture

            if pose.get('lying_down', False):
                score += 40  # Highly vulnerable position

            if pose.get('running', False):
                score += 25  # Possibly fleeing from threat

            if pose.get('crouching', False):
                score += 20  # Defensive position

        return score

    def detect_alerts(self, persons, current_time=None):
        """Enhanced alert detection with risk scoring and temporal persistence"""
        if not persons:
            return False, None

        if current_time is None:
            current_time = datetime.now()

        # Count males and females
        females = [p for p in persons if p['gender'] == "Female"]
        males = [p for p in persons if p['gender'] == "Male"]

        # No females detected, reset and return
        if not females:
            self.reset_potential_alerts()
            return False, None

        alerts = []
        hour = current_time.hour
        is_night = hour >= self.alert_config['NIGHT_HOURS_START'] or hour < self.alert_config['NIGHT_HOURS_END']

        # Context factors - check if this is a likely safe context
        # If large group with mixed genders, likely safer
        is_safe_social_context = len(persons) > 5 and len(
            females) >= 2 and len(males) >= 2

        # Process each female to calculate individual risks
        for female_idx, female in enumerate(females):
            # Skip if in likely safe social context
            if is_safe_social_context:
                continue

            # Calculate risk score for this female
            risk_score = self.calculate_risk_score(
                female, females, males, is_night, persons)

            # Store risk score for display
            female['risk_score'] = risk_score

            # Track potential alert over time
            if 'track_id' in female and risk_score > self.alert_config['RISK_THRESHOLD']:
                track_key = f"risk_{female['track_id']}"

                # Increment counter for consecutive risky frames
                if track_key not in self.potential_alerts:
                    self.potential_alerts[track_key] = 1
                else:
                    self.potential_alerts[track_key] += 1

                # Alert if risk persists for multiple frames
                if self.potential_alerts[track_key] >= self.alert_config['FRAME_THRESHOLD']:
                    alert_message = self.generate_alert_message(
                        female, females, males, is_night, risk_score)

                    alerts.append({
                        'message': alert_message[0],
                        'level': alert_message[1],
                        'risk_score': risk_score
                    })

            # Clean up if risk has decreased
            elif 'track_id' in female:
                track_key = f"risk_{female['track_id']}"
                if track_key in self.potential_alerts:
                    del self.potential_alerts[track_key]

        # Clean potential alerts for tracks that no longer exist
        self.clean_potential_alerts(persons)

        # Get highest severity alert
        if alerts:
            # Sort by alert level (higher is more severe)
            alerts.sort(key=lambda x: (
                x['level'], x['risk_score']), reverse=True)
            highest_alert = alerts[0]

            self.log_message(
                f"Alert detected: {highest_alert['message']} (Risk: {highest_alert['risk_score']})")
            return True, highest_alert['message']

        return False, None

    def filter_valid_detections(self, detections):
        """Filter out detections that are too small"""
        valid_detections = []
        for box in detections:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            # Filter out detections that are too small
            if width > self.MINIMUM_PERSON_SIZE and height > self.MINIMUM_PERSON_SIZE:
                valid_detections.append(box)

        return valid_detections

    def update_person_trackers(self, detections, frame):
        """Update person trackers with new detections"""
        # Skip if no detections
        if not detections:
            return

        # Mark all current tracks as unmatched initially
        unmatched_trackers = list(self.person_trackers.keys())
        matched_detections = []

        # Match detections to existing trackers
        for detection_box in detections:
            best_match_id = None
            best_match_distance = float('inf')

            # Find closest tracker
            for track_id in unmatched_trackers:
                tracker = self.person_trackers[track_id]
                distance = self.center_distance(detection_box, tracker['bbox'])

                if distance < self.MAX_TRACK_DISTANCE and distance < best_match_distance:
                    best_match_distance = distance
                    best_match_id = track_id

            # Update matched tracker
            if best_match_id is not None:
                self.person_trackers[best_match_id]['bbox'] = detection_box
                self.person_trackers[best_match_id]['ttl'] = self.TRACK_EXPIRATION
                unmatched_trackers.remove(best_match_id)
                matched_detections.append(detection_box)

            # Create new tracker for unmatched detection
            else:
                # Extract person image for gender analysis
                x1, y1, x2, y2 = detection_box
                person_img = frame[y1:y2,
                                   x1:x2] if y2 > y1 and x2 > x1 else None

                gender, confidence = "Unknown", 0.0

                if person_img is not None and person_img.size > 0:
                    gender, confidence = self.predict_gender(person_img)

                self.person_trackers[self.next_track_id] = {
                    'bbox': detection_box,
                    'ttl': self.TRACK_EXPIRATION,
                    'gender': gender,
                    'gender_confidence': confidence,
                    'last_gender_update': time.time()
                }

                self.next_track_id += 1

        # Update TTL for unmatched trackers
        # Use list() to safely modify during iteration
        for track_id in list(unmatched_trackers):
            self.person_trackers[track_id]['ttl'] -= 1

            # Remove expired trackers
            if self.person_trackers[track_id]['ttl'] <= 0:
                del self.person_trackers[track_id]

    def reset_potential_alerts(self):
        """Reset all potential alerts"""
        self.potential_alerts = {}

    def clean_potential_alerts(self, persons):
        """Remove potential alerts for persons no longer in the scene"""
        tracked_ids = set(p.get('track_id')
                          for p in persons if 'track_id' in p)

        keys_to_remove = []
        for key in self.potential_alerts:
            # Keys are in format "risk_trackid"
            if key.startswith("risk_"):
                track_id = int(key.split('_')[1])
                if track_id not in tracked_ids:
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.potential_alerts[key]

    def generate_alert_message(self, female, females, males, is_night, risk_score):
        """Generate appropriate alert message based on risk factors"""
        alert_message = ""
        alert_level = self.alert_levels["NOTICE"]  # Default lowest level

        # Determine most significant factor for the alert
        if 'pose' in female:
            pose = female['pose']

            if pose.get('lying_down', False):
                alert_message = "⚠️ POTENTIAL EMERGENCY: Female in vulnerable lying position"
                alert_level = self.alert_levels["ALERT"]

            elif pose.get('hands_up', False):
                alert_message = "⚠️ Woman may be in defensive posture"
                alert_level = self.alert_levels["WARNING"]

            elif pose.get('running', False):
                alert_message = "⚠️ Female appears to be running"
                alert_level = self.alert_levels["WARNING"]

        if not alert_message:
            # Count males in close proximity
            proximity_males = 0
            f_x1, f_y1, f_x2, f_y2 = female['bbox']
            f_center_x = (f_x1 + f_x2) / 2
            f_center_y = (f_y1 + f_y2) / 2

            for male in males:
                m_x1, m_y1, m_x2, m_y2 = male['bbox']
                m_center_x = (m_x1 + m_x2) / 2
                m_center_y = (m_y1 + m_y2) / 2

                distance = np.sqrt(
                    (f_center_x - m_center_x)**2 + (f_center_y - m_center_y)**2
                )

                if distance < self.alert_config['PROXIMITY_THRESHOLD']:
                    proximity_males += 1            # Alert based on male proximity
            # Count nearby females to evaluate context
            nearby_females = 0
            for other_female in females:
                if other_female is female:  # Skip the current female
                    continue

                of_x1, of_y1, of_x2, of_y2 = other_female['bbox']
                of_center_x = (of_x1 + of_x2) / 2
                of_center_y = (of_y1 + of_y2) / 2

                distance = np.sqrt(
                    (f_center_x - of_center_x)**2 +
                    (f_center_y - of_center_y)**2
                )

                if distance < self.alert_config['PROXIMITY_THRESHOLD']:
                    nearby_females += 1

            # Alert logic that considers the ratio of males to females
            if proximity_males >= 3 and proximity_males > nearby_females:
                alert_message = "⚠️ Female surrounded by multiple males"
                alert_level = self.alert_levels["WARNING"]

            elif proximity_males >= 2 and proximity_males > nearby_females:
                alert_message = "⚠️ Female approached by multiple males"
                alert_level = self.alert_levels["NOTICE"]

            elif proximity_males == 1 and is_night and nearby_females == 0:
                alert_message = "⚠️ Female approached by male at night"
                alert_level = self.alert_levels["NOTICE"]

        # If still no message but high risk score
        if not alert_message and risk_score > self.alert_config['RISK_THRESHOLD']:
            alert_message = "⚠️ Potential safety concern detected"
            alert_level = self.alert_levels["NOTICE"]

        return alert_message, alert_level

    def add_risk_indicators(self, frame, females, males, is_night, persons_detected):
        """Add visual indicators of risk level to the frame"""
        # Colored circle indicators for risk levels
        for female in females:
            if 'risk_score' in female:
                risk_score = female['risk_score']
                x1, y1, x2, y2 = female['bbox']

                # Color based on risk level
                if risk_score > 80:
                    color = (0, 0, 255)  # Red (high risk)
                elif risk_score > 50:
                    color = (0, 165, 255)  # Orange (medium risk)
                else:
                    color = (0, 255, 0)  # Green (low risk)

                # Draw risk indicator
                cv2.circle(frame, (x1 + 10, y1 + 10), 8, color, -1)

                # Add risk score if enabled
                if self.show_risk_scores:
                    cv2.putText(frame, f"Risk: {risk_score}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                color, 1, cv2.LINE_AA)

        return frame

    def process_frame(self, frame, frame_count):
        """Process a single frame of video"""
        # Copy of the frame for display
        result_frame = frame.copy()

        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]

        # Process frame with YOLO
        results = self.model(frame, conf=self.DETECTION_CONFIDENCE, classes=0)

        # Extract YOLO detections
        yolo_boxes = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates in (x1, y1, x2, y2) format
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                yolo_boxes.append((x1, y1, x2, y2))

                # Draw detection box on result frame
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Filter out small detections
        valid_detections = self.filter_valid_detections(yolo_boxes)

        # Apply non-maximum suppression to remove overlapping boxes
        filtered_detections = self.non_max_suppression(
            valid_detections, iou_threshold=0.4)

        # Update person trackers with filtered detections
        self.update_person_trackers(filtered_detections, frame)

        # Process each tracked person
        persons_detected = []

        # Gender re-evaluation settings
        GENDER_UPDATE_INTERVAL = 2.0  # Re-evaluate gender every 2 seconds

        for track_id, tracker in list(self.person_trackers.items()):
            # Extract info from tracker
            bbox = tracker['bbox']
            x1, y1, x2, y2 = bbox
            gender = tracker['gender']
            gender_confidence = tracker['gender_confidence']

            # Check if time to update gender
            should_update_gender = (
                time.time() - tracker.get('last_gender_update', 0) > GENDER_UPDATE_INTERVAL
            )

            # Get person region for further processing
            person_img = None
            if y1 < y2 and x1 < x2 and y1 >= 0 and x1 >= 0:
                if y2 < frame.shape[0] and x2 < frame.shape[1]:
                    person_img = frame[y1:y2, x1:x2]

            # Update gender if needed
            if should_update_gender and person_img is not None and person_img.size > 0:
                # Make the gender classification cache key more robust
                person_hash = hash(track_id)

                # Check if we have a cached gender result
                if person_hash in self.person_gender_cache:
                    gender, gender_confidence = self.person_gender_cache[person_hash]
                else:
                    # Re-evaluate gender and update cache
                    gender, gender_confidence = self.predict_gender(person_img)
                    self.person_gender_cache[person_hash] = (
                        gender, gender_confidence)

                # Update tracker
                tracker['gender'] = gender
                tracker['gender_confidence'] = gender_confidence
                tracker['last_gender_update'] = time.time()

            # Process pose if person region is valid
            pose_landmarks, pose_analysis = None, None
            if person_img is not None and person_img.size > 0:
                pose_landmarks, pose_analysis = self.pose_analyzer.detect_pose(
                    frame, bbox)

                # Draw pose if available
                if pose_landmarks:
                    result_frame = self.pose_analyzer.draw_pose(
                        result_frame, pose_landmarks)

            # Draw person tracker info
            label_color = (0, 165, 255) if gender == "Male" else (255, 0, 255)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), label_color, 2)

            gender_label = f"{gender} ({gender_confidence:.2f})"
            cv2.putText(result_frame, f"ID:{track_id} {gender_label}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        label_color, 1, cv2.LINE_AA)

            # Add to person detections
            person_info = {
                'track_id': track_id,
                'bbox': bbox,
                'gender': gender,
                'gender_confidence': gender_confidence
            }

            # Add pose information if available
            if pose_analysis:
                person_info['pose'] = pose_analysis

                # Draw pose labels if any pose detected
                pose_text = []
                if pose_analysis.get('hands_up', False):
                    pose_text.append("Hands Up")
                if pose_analysis.get('lying_down', False):
                    pose_text.append("Lying Down")
                if pose_analysis.get('running', False):
                    pose_text.append("Running")
                if pose_analysis.get('crouching', False):
                    pose_text.append("Crouching")

                if pose_text:
                    cv2.putText(result_frame, ", ".join(pose_text),
                                (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, label_color, 1, cv2.LINE_AA)

            persons_detected.append(person_info)

        # Check for alerts
        alert_detected, alert_message = self.detect_alerts(persons_detected)

        # Apply alert cooldown
        if alert_detected and time.time() - self.last_alert_time > self.alert_config['ALERT_COOLDOWN']:
            self.last_alert_time = time.time()
            # Add large alert message
            cv2.putText(result_frame, alert_message,
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2, cv2.LINE_AA)

            # Save evidence image
            evidence_path = os.path.join(
                "evidence", f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(evidence_path, result_frame)
            self.log_message(f"Alert saved to {evidence_path}")
        elif alert_detected:
            # Show alert but with cooldown
            cv2.putText(result_frame, alert_message,
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 165, 255), 2, cv2.LINE_AA)

        # Clean up cache if it gets too large
        if len(self.person_gender_cache) > 100:
            self.person_gender_cache = {}

        # Add stats display
        current_time = datetime.now()
        hour = current_time.hour
        is_night = hour >= self.alert_config['NIGHT_HOURS_START'] or hour < self.alert_config['NIGHT_HOURS_END']

        # Define females and males AFTER persons_detected is populated
        females = [p for p in persons_detected if p['gender'] == "Female"]
        males = [p for p in persons_detected if p['gender'] == "Male"]

        stats_text = [
            f"Frame: {frame_count}",
            f"Persons: {len(persons_detected)}",
            f"Female: {len(females)}",
            f"Male: {len(males)}",
            f"Time: {current_time.strftime('%H:%M:%S')}"
        ]

        for i, text in enumerate(stats_text):
            cv2.putText(result_frame, text,
                        (10, frame_height - 20 - i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

        # Add risk indicators
        result_frame = self.add_risk_indicators(
            result_frame, females, males, is_night, persons_detected)

        # Return annotated frame
        return result_frame

    def run_detection(self):
        """Main function to run the detection pipeline on video"""
        # Connect to camera
        cap = self.connect_camera()

        frame_count = 0
        alert_sound_played = False
        self.log_message("Starting detection loop")

        while cap.isOpened():
            # Read frame
            ret, frame = cap.read()

            if not ret:
                print("Failed to read frame")
                self.log_message(
                    "Failed to read frame, attempting reconnection")
                # Try to reconnect
                cap.release()
                time.sleep(1)
                cap = self.connect_camera()
                continue

            frame_count += 1

            # Skip frames for performance if needed
            # Process every frame (change for performance)
            if frame_count % 1 != 0:
                continue

            # Process frame
            try:
                result_frame = self.process_frame(frame, frame_count)

                # Show result
                cv2.imshow('StreeRaksha Safety Monitoring', result_frame)

                # Check for exit
                key = cv2.waitKey(1)
                if key == 27 or key == ord('q'):  # ESC or 'q' key
                    break

            except Exception as e:
                print(f"Error processing frame: {e}")
                self.log_message(f"Error processing frame: {e}")

        # Release resources
        self.log_message("Shutting down")
        cap.release()
        cv2.destroyAllWindows()


# Main execution
if __name__ == "__main__":
    print("This is a module file. Run streeraksha.py to start the application.")
