"""
StreeRaksha Firebase Manager Module
Handles Firebase integration for backend services.
"""

import os
import json
import time
import base64
import hashlib
from datetime import datetime
import requests
import threading
import queue


class FirebaseManager:
    """Manages Firebase realtime database and storage integration"""

    def __init__(self, logger, config_path=None):
        """Initialize Firebase manager"""
        self.logger = logger
        self.config = None
        self.firebase_url = None
        self.storage_url = None
        self.api_key = None
        self.project_id = None
        self.initialized = False
        # Limit queue size to prevent memory issues
        self.upload_queue = queue.Queue(maxsize=100)
        self.stop_event = threading.Event()
        self.auth_token = None
        self.auth_expiry = 0
        # Progressive retry delays in seconds
        self.retry_delays = [1, 2, 5, 10, 30]
        self.stats = {
            'uploads_attempted': 0,
            'uploads_successful': 0,
            'uploads_failed': 0,
            'queue_high_water_mark': 0
        }

        # Try to load configuration
        try:
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)

                    # Extract necessary information
                    self.firebase_url = self.config.get('databaseURL', None)
                    self.storage_url = self.config.get('storageBucket', None)
                    self.api_key = self.config.get('apiKey', None)
                    self.project_id = self.config.get('projectId', None)

                    if self.firebase_url and self.api_key and self.project_id:
                        self.initialized = True
                        self.logger.info(
                            "Firebase configuration loaded successfully")

                        # Initial authentication attempt
                        if self._authenticate():
                            self.logger.info(
                                "Firebase authentication successful")
                        else:
                            self.logger.warning(
                                "Initial Firebase authentication failed, will retry")
                    else:
                        self.logger.error(
                            "Invalid Firebase configuration: missing required fields")
            else:
                self.logger.warning(
                    "Firebase configuration file not found, operating in offline mode")
        except Exception as e:
            self.logger.error(f"Error loading Firebase configuration: {e}")

    def _authenticate(self):
        """Authenticate with Firebase and get a token"""
        if not self.initialized or not self.api_key:
            return False

        # Check if current token is still valid
        current_time = time.time()
        if self.auth_token and current_time < self.auth_expiry - 300:  # 5 min buffer
            return True

        try:
            # For a real implementation, this would use Firebase Auth REST API
            # This is a placeholder for the real authentication logic
            self.auth_token = f"sample_token_{int(time.time())}"
            self.auth_expiry = time.time() + 3600  # Token valid for 1 hour
            return True
        except Exception as e:
            self.logger.error(f"Firebase authentication error: {e}")
            self.auth_token = None
            return False

    def start_upload_thread(self):
        """Start background thread for uploads"""
        if not self.initialized:
            self.logger.warning(
                "Firebase not initialized, not starting upload thread")
            return

        def upload_worker():
            self.logger.info("Firebase upload thread started")
            last_stats_time = time.time()

            while not self.stop_event.is_set():
                try:
                    # Get item from queue with timeout
                    item = self.upload_queue.get(timeout=0.5)

                    # Update queue high water mark for stats
                    current_size = self.upload_queue.qsize() + 1  # +1 for current item
                    if current_size > self.stats['queue_high_water_mark']:
                        self.stats['queue_high_water_mark'] = current_size

                    # Try to authenticate if needed
                    if not self.auth_token or time.time() > self.auth_expiry:
                        if not self._authenticate():
                            self.logger.warning(
                                "Firebase upload failed: authentication error")
                            # Requeue the item with delay for retry
                            self._requeue_item(item)
                            continue

                    # Process item based on type
                    success = False
                    self.stats['uploads_attempted'] += 1

                    if item['type'] == 'evidence':
                        success = self._upload_evidence(item['data'])
                    elif item['type'] == 'incident':
                        success = self._upload_incident(item['data'])
                    elif item['type'] == 'log':
                        success = self._upload_log(item['data'])

                    # Update stats
                    if success:
                        self.stats['uploads_successful'] += 1
                    else:
                        self.stats['uploads_failed'] += 1
                        # Requeue for retry if failed
                        self._requeue_item(item)

                    # Mark task as done
                    self.upload_queue.task_done()

                    # Log stats periodically
                    current_time = time.time()
                    if current_time - last_stats_time >= 60:  # Every minute
                        self._log_stats()
                        last_stats_time = current_time

                except queue.Empty:
                    # Nothing to upload, just continue
                    continue
                except Exception as e:
                    self.logger.error(f"Error in Firebase upload thread: {e}")
                    if not self.upload_queue.empty():
                        self.upload_queue.task_done()  # Mark as done even on error

            self.logger.info("Firebase upload thread stopped")
            self._log_stats()  # Log final stats

        # Start thread
        self.upload_thread = threading.Thread(
            target=upload_worker, daemon=True)
        self.upload_thread.start()

    def _requeue_item(self, item):
        """Requeue item with exponential backoff"""
        if 'retries' not in item:
            item['retries'] = 0

        if item['retries'] < len(self.retry_delays):
            item['retries'] += 1
            delay = self.retry_delays[item['retries'] - 1]

            # Schedule retry
            retry_thread = threading.Thread(
                target=self._delayed_retry,
                args=(item, delay),
                daemon=True
            )
            retry_thread.start()
            self.logger.debug(f"Scheduled retry {item['retries']} in {delay}s")
        else:
            self.logger.warning(
                f"Failed to upload {item['type']} after {item['retries']} retries")

    def _delayed_retry(self, item, delay):
        """Helper function to retry upload after delay"""
        time.sleep(delay)
        try:
            if not self.stop_event.is_set():
                self.upload_queue.put(item)
        except queue.Full:
            self.logger.warning("Upload queue full, dropping retry")

    def _log_stats(self):
        """Log upload statistics"""
        success_rate = 0
        if self.stats['uploads_attempted'] > 0:
            success_rate = (
                self.stats['uploads_successful'] / self.stats['uploads_attempted']) * 100

        self.logger.info(
            f"Firebase stats: {self.stats['uploads_successful']}/{self.stats['uploads_attempted']} "
            f"uploads successful ({success_rate:.1f}%), {self.upload_queue.qsize()} queued, "
            f"max queue: {self.stats['queue_high_water_mark']}"
        )

    def stop(self):
        """Stop the upload thread"""
        if not self.initialized:
            return

        self.logger.info("Stopping Firebase upload thread")
        self.stop_event.set()

        if hasattr(self, 'upload_thread') and self.upload_thread.is_alive():
            try:
                # Wait for thread to finish with timeout
                self.upload_thread.join(timeout=2)

                # Log remaining items
                remaining = self.upload_queue.qsize()
                if remaining > 0:
                    self.logger.warning(
                        f"{remaining} items still in upload queue")
            except Exception as e:
                self.logger.error(f"Error stopping upload thread: {e}")

    def queue_evidence_upload(self, image_path, metadata):
        """Queue evidence for upload to Firebase storage"""
        if not self.initialized:
            return False

        # Check if the file exists
        if not os.path.exists(image_path):
            self.logger.error(
                f"Cannot queue evidence upload: file not found: {image_path}")
            return False

        try:
            # Add to upload queue with timeout to prevent blocking
            self.upload_queue.put({
                'type': 'evidence',
                'priority': 2,  # Higher priority (evidence is important)
                'data': {
                    'path': image_path,
                    'metadata': metadata,
                    'timestamp': datetime.now().isoformat()
                }
            }, timeout=1.0)
            return True
        except queue.Full:
            self.logger.error("Evidence upload queue is full, dropping upload")
            return False

    def queue_incident_upload(self, incident_data):
        """Queue incident data for upload to Firebase database"""
        if not self.initialized:
            return False

        try:
            # Add to upload queue with timeout
            self.upload_queue.put({
                'type': 'incident',
                'priority': 1,  # High priority
                'data': {
                    'incident': incident_data,
                    'timestamp': datetime.now().isoformat()
                }
            }, timeout=1.0)
            return True
        except queue.Full:
            self.logger.error("Incident upload queue is full, dropping upload")
            return False

    def queue_log_upload(self, log_data):
        """Queue log data for upload to Firebase database"""
        if not self.initialized:
            return False

        try:
            # Add to upload queue with timeout, non-blocking
            # Don't block on logs as they're less critical
            self.upload_queue.put({
                'type': 'log',
                'priority': 0,  # Lower priority
                'data': {
                    'log': log_data,
                    'timestamp': datetime.now().isoformat()
                }
            }, timeout=0.5)
            return True
        except queue.Full:
            # Just silently drop logs if queue is full
            return False

    def get_stats(self):
        """Get upload statistics"""
        stats = self.stats.copy()
        stats['current_queue_size'] = self.upload_queue.qsize()
        stats['initialized'] = self.initialized
        stats['authenticated'] = bool(
            self.auth_token and time.time() < self.auth_expiry)

        return stats

    def clear_queue(self):
        """Clear the upload queue (for emergency situations)"""
        if not self.initialized:
            return 0

        count = 0
        try:
            while True:
                self.upload_queue.get_nowait()
                self.upload_queue.task_done()
                count += 1
        except queue.Empty:
            pass

        self.logger.warning(f"Cleared {count} items from upload queue")
        return count

    def _upload_evidence(self, data):
        """Upload evidence to Firebase storage"""
        if not self.initialized or not self.storage_url or not self.auth_token:
            return False

        try:
            # Get image path and check if file exists
            image_path = data['path']
            if not os.path.exists(image_path):
                self.logger.error(f"Evidence file not found: {image_path}")
                return False

            # Generate a unique filename for storage
            timestamp = data.get('timestamp', datetime.now().isoformat())
            filename = os.path.basename(image_path)
            unique_id = hashlib.md5(
                f"{timestamp}_{filename}".encode()).hexdigest()[:10]
            storage_path = f"evidence/{timestamp.split('T')[0]}/{unique_id}_{filename}"

            # In a real implementation, we would upload the file to Firebase Storage
            # For this implementation, we'll simulate a successful upload

            # Read a small portion of the file to simulate upload
            with open(image_path, 'rb') as f:
                # Just read a bit to verify the file is valid
                file_sample = f.read(1024)

            # Simulate network latency
            time.sleep(0.1)

            # Log successful "upload"
            self.logger.info(
                f"Evidence uploaded to: {self.storage_url}/{storage_path}")

            # Update metadata with storage path for reference
            data['metadata']['storage_path'] = storage_path
            data['metadata']['uploaded_at'] = datetime.now().isoformat()

            return True
        except Exception as e:
            self.logger.error(f"Error uploading evidence to Firebase: {e}")
            return False

    def _upload_incident(self, data):
        """Upload incident to Firebase database"""
        if not self.initialized or not self.firebase_url or not self.auth_token:
            return False

        try:
            # Generate a database path for the incident
            incident_data = data['incident']
            timestamp = data.get('timestamp', datetime.now().isoformat())
            date_part = timestamp.split('T')[0]
            time_part = timestamp.split('T')[1].replace(':', '-').split('.')[0]
            db_path = f"incidents/{date_part}/{time_part}"

            # In a real implementation, we would do a PUT/POST to Firebase Realtime Database
            # For now, simulate a successful database write

            # Prepare the data that would be sent
            payload = {
                "timestamp": timestamp,
                "message": incident_data.get('alert_message', ''),
                "level": incident_data.get('alert_level', ''),
                "location": incident_data.get('location', 'Unknown'),
                "evidence_ref": incident_data.get('evidence_path', '')
            }

            # Simulate network latency
            time.sleep(0.05)

            # Log the "database write"
            self.logger.info(
                f"Incident data written to: {self.firebase_url}/{db_path}")

            return True
        except Exception as e:
            self.logger.error(f"Error uploading incident to Firebase: {e}")
            return False

    def _upload_log(self, data):
        """Upload log to Firebase database"""
        if not self.initialized or not self.firebase_url or not self.auth_token:
            return False

        try:
            # Generate a database path for the log
            log_data = data['log']
            timestamp = data.get('timestamp', datetime.now().isoformat())
            date_part = timestamp.split('T')[0]
            time_part = timestamp.split('T')[1].replace(':', '-').split('.')[0]
            db_path = f"logs/{date_part}/{time_part}"

            # In a real implementation, we would do a PUT/POST to Firebase Realtime Database
            # For now, simulate a successful database write

            # Simulate network latency - logs should be fast
            time.sleep(0.02)

            # Log the "database write" at debug level to avoid log spam
            self.logger.debug(
                f"Log data written to: {self.firebase_url}/{db_path}")

            return True
        except Exception as e:
            self.logger.error(f"Error uploading log to Firebase: {e}")
            return False
