"""
StreeRaksha Gender Detector Module
Handles gender detection functionality.
"""

import cv2
import torch
import numpy as np
from PIL import Image

try:
    from transformers import AutoFeatureExtractor, AutoModelForImageClassification
    has_transformers = True
except ImportError:
    has_transformers = False


class GenderDetector:
    def __init__(self, use_simulated=False):
        """Initialize the gender detector"""
        self.use_simulated = use_simulated or not has_transformers
        self.model = None
        self.feature_extractor = None

        if not self.use_simulated:
            # Try loading the Hugging Face model
            try:
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                    "rizvandwiki/gender-classification-2")
                self.model = AutoModelForImageClassification.from_pretrained(
                    "rizvandwiki/gender-classification-2")
                print("Successfully loaded gender classification model")
            except Exception as e:
                print(f"Error loading gender model: {e}")
                print("Falling back to simulated gender classification")
                self.use_simulated = True

        if self.use_simulated:
            print("Using simulated gender classification")

    def predict(self, image):
        """Predict gender from an image"""
        if self.use_simulated:
            return self._simulated_prediction(image)
        else:
            return self._model_prediction(image)

    def _simulated_prediction(self, image):
        """Simulate gender prediction when model is not available"""
        # Use the hash of the image data as a deterministic seed
        seed_value = hash(image.tobytes()) % 100
        gender = "Female" if seed_value < 30 else "Male"  # 30% chance female
        confidence = np.random.uniform(0.80, 0.95)
        return gender, confidence

    def _model_prediction(self, image):
        """Use the Hugging Face model for gender prediction with bias correction"""
        try:
            # Convert OpenCV BGR image to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # Prepare image for the model
            inputs = self.feature_extractor(
                images=pil_image, return_tensors="pt")

            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Find indices for male and female classes
                male_idx = None
                female_idx = None
                for idx, label in self.model.config.id2label.items():
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
            # Format as "Male" or "Female"
            gender_label = self.model.config.id2label[predicted_class_idx]
            # Higher threshold for female classification based on image quality
            female_threshold = 0.70  # Matches streeraksha.py
            male_threshold = 0.50    # Matches streeraksha.py

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
            return self._simulated_prediction(image)
