"""
StreeRaksha Alert System Module
Handles risk assessment and alert generation.
"""

from datetime import datetime
import os
import cv2
import time
import numpy as np
import json


class AlertSystem:
    def __init__(self, firebase_manager=None):
        """Initialize the alert system"""
        # Alert configuration
        self.alert_config = {
            'NIGHT_HOURS_START': 18,
            'NIGHT_HOURS_END': 6,
            'PROXIMITY_THRESHOLD': 150,
            'ALERT_COOLDOWN': 10.0,
            'RISK_THRESHOLD': 60,     # Minimum risk score to trigger alert
            'FRAME_THRESHOLD': 5      # Number of consecutive frames before alert
        }

        # Firebase manager for incident upload
        self.firebase_manager = firebase_manager

        # Alert levels
        self.alert_levels = {
            "NOTICE": 1,    # Might be concerning
            "WARNING": 2,   # Potentially troubling
            "ALERT": 3      # Serious concern
        }

        # Potential alerts storage
        self.potential_alerts = {}  # Track {alert_type: consecutive_frames}

        # Last alert time for cooldown
        self.last_alert_time = 0

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
                        'risk_score': risk_score,
                        'female': female
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
            return True, highest_alert['message']

        return False, None

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

    def add_risk_indicators(self, frame, females, males, is_night, show_risk_scores=True):
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
                if show_risk_scores:
                    cv2.putText(frame, f"Risk: {risk_score}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                color, 1, cv2.LINE_AA)

        return frame

    def save_evidence(self, frame, alert_message=None, alert_level=None):
        """Save evidence frame to disk and optionally upload to Firebase"""
        # Create directory if it doesn't exist
        os.makedirs("evidence", exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now()
        evidence_filename = f"alert_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        evidence_path = os.path.join("evidence", evidence_filename)

        # Save image to disk
        cv2.imwrite(evidence_path, frame)

        # Upload to Firebase if manager is available
        if self.firebase_manager:
            # Prepare metadata for the evidence
            metadata = {
                "timestamp": timestamp.isoformat(),
                "alert_message": alert_message,
                "alert_level": alert_level,
                "location": "Camera 1",  # This would be dynamic in a real setup
                "filename": evidence_filename
            }

            # Queue for upload
            self.firebase_manager.queue_evidence_upload(
                evidence_path, metadata)

            # Also upload incident data
            incident_data = {
                "timestamp": timestamp.isoformat(),
                "alert_message": alert_message,
                "alert_level": alert_level,
                "evidence_path": evidence_path,
                "location": "Camera 1"
            }
            self.firebase_manager.queue_incident_upload(incident_data)

        return evidence_path

    def check_cooldown(self):
        """Check if the alert system is in cooldown"""
        return time.time() - self.last_alert_time <= self.alert_config['ALERT_COOLDOWN']

    def update_cooldown(self):
        """Update the last alert time"""
        self.last_alert_time = time.time()
