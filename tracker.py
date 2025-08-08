"""
StreeRaksha Tracker Module
Handles person tracking functionality.
"""

import time
import numpy as np


class PersonTracker:
    def __init__(self, track_expiration=8, max_track_distance=50):
        """Initialize the person tracker"""
        self.trackers = {}
        self.next_track_id = 0
        self.TRACK_EXPIRATION = track_expiration
        self.MAX_TRACK_DISTANCE = max_track_distance
        self.MIN_SIZE = 60  # Minimum person size to match the original StreeRakshaDetector

        # Gender reclassification parameters
        # Standard interval: Re-classify gender every 3 seconds
        self.GENDER_REFRESH_INTERVAL = 3.0
        # Confidence threshold: Re-classify sooner if confidence is below this
        self.GENDER_REFRESH_THRESHOLD = 0.8
        # Multiplier for low confidence refresh (half the normal interval)
        self.GENDER_REFRESH_LOW_CONF_MULTIPLIER = 0.5
        self.GENDER_MIN_CONFIDENCE = 0.6  # Minimum confidence required to update gender

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

    # Use 0.4 like the original code
    def non_max_suppression(self, boxes, iou_threshold=0.4):
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

    def filter_valid_detections(self, detections, min_size=60):
        """Filter out detections that are too small"""
        valid_detections = []
        for box in detections:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            # Filter out detections that are too small
            if width > min_size and height > min_size:
                valid_detections.append(box)

        return valid_detections

    def update_trackers(self, detections, frame, gender_detector=None):
        """Update person trackers with new detections"""
        # Skip if no detections but still update TTL and return existing trackers
        if not detections:
            # Update TTL for all trackers
            for track_id in list(self.trackers.keys()):
                self.trackers[track_id]['ttl'] -= 1
                if self.trackers[track_id]['ttl'] <= 0:
                    del self.trackers[track_id]

            # Return remaining trackers
            return self._get_tracked_persons()

        # Mark all current tracks as unmatched initially
        unmatched_trackers = list(self.trackers.keys())
        matched_detections = []

        # Match detections to existing trackers - exact algorithm from the original streeraksha.py
        for detection_box in detections:
            # Minimum IoU threshold (higher than default for better stability)
            best_iou = 0.2
            best_match_id = None
            best_match_distance = float('inf')

            for track_id in unmatched_trackers:
                tracker = self.trackers[track_id]
                track_box = tracker['bbox']

                # Calculate both IoU and center distance
                iou = self.calculate_iou(detection_box, track_box)
                distance = self.center_distance(detection_box, track_box)

                # More aggressive matching logic to ensure stable tracking:
                # 1. Good IoU is the primary criteria (boxes overlapping significantly)
                # 2. Reasonable IoU with close distance (continuing movement)
                # 3. Very close centers even with low IoU (fast movements)
                if (iou > best_iou or
                    (iou > 0.1 and distance < self.MAX_TRACK_DISTANCE) or
                        (distance < self.MAX_TRACK_DISTANCE * 0.4 and iou > 0.05)):
                    best_iou = iou
                    best_match_id = track_id
                    best_match_distance = distance            # Update matched tracker
            if best_match_id is not None:
                tracker = self.trackers[best_match_id]
                tracker['bbox'] = detection_box
                tracker['ttl'] = self.TRACK_EXPIRATION
                # Enhanced periodic gender re-classification:
                # Check if we should re-classify gender based on time and previous confidence
                current_time = time.time()
                time_since_update = current_time - \
                    tracker.get('last_gender_update', 0)
                gender_confidence = tracker.get('gender_confidence', 0.0)

                # Determine if re-classification is needed:
                # 1. Standard interval has passed
                # 2. Low confidence in current gender + half the standard interval has passed
                # 3. Person has changed significantly (area/proportion change > 20%)
                should_refresh = (
                    time_since_update > self.GENDER_REFRESH_INTERVAL or
                    (gender_confidence < self.GENDER_REFRESH_THRESHOLD and
                     time_since_update > self.GENDER_REFRESH_INTERVAL * self.GENDER_REFRESH_LOW_CONF_MULTIPLIER)
                )

                if should_refresh and gender_detector is not None:
                    x1, y1, x2, y2 = detection_box
                    # Extract person image for updated gender analysis
                    person_img = frame[y1:y2,
                                       x1:x2] if y2 > y1 and x2 > x1 else None

                    if person_img is not None and person_img.size > 0:
                        new_gender, new_confidence = gender_detector.predict(
                            person_img)

                        # Track confidence history for more stable classifications
                        old_gender = tracker.get('gender', 'Unknown')
                        old_confidence = tracker.get('gender_confidence', 0.0)

                        # Only update if the new classification meets our confidence threshold
                        # or if it's the same gender as before with reasonable confidence
                        if (new_confidence > self.GENDER_MIN_CONFIDENCE or
                                (new_gender == old_gender and new_confidence > old_confidence * 0.8)):

                            # Apply hysteresis to prevent gender flip-flopping
                            # If changing from one gender to another, require higher confidence
                            if new_gender != old_gender and old_confidence > 0.7:
                                # Need 20% higher confidence to change an established gender
                                confidence_threshold = old_confidence * 1.2
                                if new_confidence > confidence_threshold:
                                    tracker['gender'] = new_gender
                                    tracker['gender_confidence'] = new_confidence
                            else:
                                tracker['gender'] = new_gender
                                tracker['gender_confidence'] = new_confidence

                            tracker['last_gender_update'] = current_time
                            tracker['gender_history'] = tracker.get(
                                'gender_history', []) + [(new_gender, new_confidence)]
                            # Keep only the last 5 classifications
                            if len(tracker['gender_history']) > 5:
                                tracker['gender_history'] = tracker['gender_history'][-5:]

                unmatched_trackers.remove(best_match_id)
                matched_detections.append(detection_box)
            else:
                # Create new tracker for unmatched detection                # Extract person image for gender analysis
                x1, y1, x2, y2 = detection_box
                person_img = frame[y1:y2,
                                   x1:x2] if y2 > y1 and x2 > x1 else None

                gender, confidence = "Unknown", 0.0

                if person_img is not None and person_img.size > 0 and gender_detector is not None:
                    gender, confidence = gender_detector.predict(person_img)

                self.trackers[self.next_track_id] = {
                    'bbox': detection_box,
                    'ttl': self.TRACK_EXPIRATION,
                    'gender': gender,
                    'gender_confidence': confidence,
                    'last_gender_update': time.time(),
                    'track_id': self.next_track_id,
                    'gender_history': [(gender, confidence)]
                }

                self.next_track_id += 1

        # Update TTL for unmatched trackers
        for track_id in list(unmatched_trackers):
            self.trackers[track_id]['ttl'] -= 1

            # Remove expired trackers
            if self.trackers[track_id]['ttl'] <= 0:
                del self.trackers[track_id]

        return self._get_tracked_persons()

    def _get_tracked_persons(self):
        """Helper method to convert tracker dictionary to person list"""
        persons = []
        for track_id, tracker in self.trackers.items():
            person_info = {
                'track_id': track_id,
                'bbox': tracker['bbox'],
                'gender': tracker['gender'],
                'gender_confidence': tracker.get('gender_confidence', 0.0),
                'ttl': tracker['ttl'],
                'gender_history': tracker.get('gender_history', []),
                'last_gender_update': tracker.get('last_gender_update', 0)
            }
            persons.append(person_info)
        return persons
