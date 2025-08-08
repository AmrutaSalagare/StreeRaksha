"""
StreeRaksha Configuration Manager Module
Handles dynamic configuration management for threads and components.
"""

import os
import json
import threading
import time
from datetime import datetime
import logging


class ConfigManager:
    """
    Manages dynamic configuration for the StreeRaksha application
    Supports runtime updates, persistence, and thread-safe access
    """

    def __init__(self, config_path=None, logger=None):
        """Initialize the configuration manager"""
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), 'config.json')
        self.logger = logger or logging.getLogger('ConfigManager')
        self._config_lock = threading.RLock()

        # Default configuration
        self._default_config = {
            "version": "1.0.0",
            "last_updated": datetime.now().isoformat(),
            "system": {
                "debug_mode": False,
                "save_evidence": True,
                "evidence_quality": 95,
                "log_level": "INFO"
            },
            "threading": {
                "frame_queue_size": 30,
                "result_queue_size": 10,
                "upload_queue_size": 100,
                "capture_thread_priority": "NORMAL",
                "processing_thread_priority": "ABOVE_NORMAL",
                "rendering_thread_priority": "NORMAL",
                "upload_thread_priority": "BELOW_NORMAL",
                "max_fps": 0  # 0 means no limit
            },
            "detection": {
                "model_confidence": 0.5,
                "person_min_size": 60,
                "nms_threshold": 0.4,
                "track_expiration": 8,
                "max_track_distance": 50
            },
            "gender": {
                "refresh_interval": 3.0,
                "refresh_threshold": 0.8,
                "min_confidence": 0.6,
                "low_confidence_multiplier": 0.5
            },
            "alerts": {
                "night_hours_start": 18,
                "night_hours_end": 6,
                "proximity_threshold": 150,
                "alert_cooldown": 10.0,
                "risk_threshold": 60,
                "frame_threshold": 5
            },
            "firebase": {
                "enabled": True,
                "retry_delays": [1, 2, 5, 10, 30],
                "max_retries": 5,
                "auth_refresh_minutes": 55
            },
            "camera": {
                "index": 0,
                "width": 640,
                "height": 480,
                "fps": 30,
                "reconnect_delay": 5.0
            },
            "ui": {
                "show_fps": True,
                "show_detections": True,
                "show_risk_scores": True,
                "show_debug_overlay": True,
                "window_title": "StreeRaksha Safety Monitoring"
            }
        }

        # Current configuration (load from file or use defaults)
        self.config = self._load_config()

        # Flag to check if config has changed
        self._config_changed = False

        # Register update callbacks
        self._update_callbacks = []

        # Start config watcher thread
        self._stop_event = threading.Event()
        self._watcher_thread = None

    def _load_config(self):
        """Load configuration from file or return defaults if file doesn't exist"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)

                # Merge with defaults to ensure all values are present
                merged_config = self._merge_configs(
                    self._default_config, loaded_config)
                self.logger.info("Configuration loaded from file")
                return merged_config
            except Exception as e:
                self.logger.error(f"Error loading configuration: {e}")
                return self._default_config.copy()
        else:
            self.logger.info("No configuration file found, using defaults")
            # Save defaults to create the file
            self._save_config(self._default_config)
            return self._default_config.copy()

    def _merge_configs(self, default, loaded):
        """Recursively merge loaded config with defaults to ensure all keys exist"""
        merged = default.copy()

        for key, value in loaded.items():
            # If the key is in default and both are dictionaries, merge them
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(default[key], value)
            else:
                # Otherwise use the loaded value
                merged[key] = value

        return merged

    def _save_config(self, config=None):
        """Save configuration to file"""
        if config is None:
            config = self.config

        try:
            # Update the last updated timestamp
            config["last_updated"] = datetime.now().isoformat()

            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)

            self.logger.info("Configuration saved to file")
            return True
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False

    def get(self, path, default=None):
        """
        Get a configuration value by path
        Path format: "section.subsection.key" e.g. "threading.frame_queue_size"
        """
        with self._config_lock:
            try:
                parts = path.split('.')
                value = self.config
                for part in parts:
                    value = value[part]
                return value
            except (KeyError, TypeError):
                return default

    def set(self, path, value):
        """
        Set a configuration value by path
        Path format: "section.subsection.key" e.g. "threading.frame_queue_size"
        """
        with self._config_lock:
            try:
                parts = path.split('.')
                config = self.config

                # Navigate to the right level
                for part in parts[:-1]:
                    if part not in config:
                        config[part] = {}
                    config = config[part]

                # Set the value
                config[parts[-1]] = value
                self._config_changed = True

                # Notify callbacks
                self._notify_callbacks(path, value)

                return True
            except Exception as e:
                self.logger.error(f"Error setting configuration value: {e}")
                return False

    def get_section(self, section):
        """Get an entire configuration section"""
        with self._config_lock:
            return self.config.get(section, {}).copy()

    def update_section(self, section, values):
        """Update an entire configuration section"""
        with self._config_lock:
            if section not in self.config:
                self.config[section] = {}

            # Update the section
            for key, value in values.items():
                self.config[section][key] = value

            self._config_changed = True

            # Notify callbacks
            for key, value in values.items():
                self._notify_callbacks(f"{section}.{key}", value)

            return True

    def save(self):
        """Save the current configuration to file"""
        with self._config_lock:
            if self._config_changed:
                success = self._save_config()
                if success:
                    self._config_changed = False
                return success
            return True

    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        with self._config_lock:
            self.config = self._default_config.copy()
            self._config_changed = True

            # Notify all callbacks
            for section, values in self.config.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        self._notify_callbacks(f"{section}.{key}", value)

            return self._save_config()

    def register_callback(self, callback, paths=None):
        """
        Register a callback function that will be called when config changes
        Callback signature: callback(path, new_value)
        Paths: optional list of paths to watch, None for all changes
        """
        self._update_callbacks.append((callback, paths))

    def unregister_callback(self, callback):
        """Unregister a previously registered callback"""
        self._update_callbacks = [
            (cb, paths) for cb, paths in self._update_callbacks if cb != callback
        ]

    def _notify_callbacks(self, path, value):
        """Notify all registered callbacks about a change"""
        for callback, paths in self._update_callbacks:
            if paths is None or path in paths:
                try:
                    callback(path, value)
                except Exception as e:
                    self.logger.error(
                        f"Error in configuration update callback: {e}")

    def start_watcher(self, interval=30):
        """Start a thread to watch for external config file changes"""
        def watch_config():
            last_modified = os.path.getmtime(
                self.config_path) if os.path.exists(self.config_path) else 0

            while not self._stop_event.is_set():
                try:
                    if os.path.exists(self.config_path):
                        current_modified = os.path.getmtime(self.config_path)

                        # Check if file was modified externally
                        if current_modified > last_modified:
                            self.logger.info(
                                "Configuration file changed externally, reloading")

                            # Load new config
                            with self._config_lock:
                                old_config = self.config.copy()
                                self.config = self._load_config()

                                # Find changed values and notify callbacks
                                self._find_and_notify_changes(
                                    old_config, self.config)

                            last_modified = current_modified

                    # Periodically save if there are pending changes
                    with self._config_lock:
                        if self._config_changed:
                            self._save_config()
                            self._config_changed = False
                            last_modified = os.path.getmtime(self.config_path)

                except Exception as e:
                    self.logger.error(f"Error in configuration watcher: {e}")

                # Wait for next check
                self._stop_event.wait(interval)

        # Start the watcher thread
        self._watcher_thread = threading.Thread(
            target=watch_config, daemon=True)
        self._watcher_thread.start()
        self.logger.info("Configuration watcher started")

    def stop_watcher(self):
        """Stop the configuration watcher thread"""
        if self._watcher_thread and self._watcher_thread.is_alive():
            self._stop_event.set()
            self._watcher_thread.join(timeout=2.0)
            self.logger.info("Configuration watcher stopped")

    def _find_and_notify_changes(self, old_config, new_config, path_prefix=""):
        """Recursively find changed values between configurations and notify callbacks"""
        for key, new_value in new_config.items():
            full_path = f"{path_prefix}.{key}" if path_prefix else key

            if key not in old_config:
                # New key added
                if isinstance(new_value, dict):
                    # For dictionaries, recurse
                    self._find_and_notify_changes({}, new_value, full_path)
                else:
                    # For values, notify directly
                    self._notify_callbacks(full_path, new_value)
            elif isinstance(new_value, dict) and isinstance(old_config[key], dict):
                # Recurse into nested dictionaries
                self._find_and_notify_changes(
                    old_config[key], new_value, full_path)
            elif new_value != old_config[key]:
                # Value changed
                self._notify_callbacks(full_path, new_value)

        # Check for deleted keys
        for key in old_config:
            if key not in new_config:
                full_path = f"{path_prefix}.{key}" if path_prefix else key
                # Notify with None value for deleted keys
                self._notify_callbacks(full_path, None)
