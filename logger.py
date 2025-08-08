"""
StreeRaksha Logger Module
Handles logging functionality.
"""

import os
from datetime import datetime


class Logger:
    def __init__(self, log_dir="logs"):
        """Initialize the logger"""
        self.log_dir = log_dir

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Create log file with timestamp
        self.log_filename = f"StreeRaksha_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.log_path = os.path.join(log_dir, self.log_filename)

        # Log initialization
        self.log("Logger initialized")

    def log(self, message, level="INFO"):
        """Log a message with specified level"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}\n"

        # Print to console
        print(f"[{level}] {message}")

        # Write to log file
        with open(self.log_path, 'a') as log_file:
            log_file.write(log_entry)

    def info(self, message):
        """Log an informational message"""
        self.log(message, "INFO")

    def warning(self, message):
        """Log a warning message"""
        self.log(message, "WARNING")

    def error(self, message):
        """Log an error message"""
        self.log(message, "ERROR")

    def debug(self, message):
        """Log a debug message"""
        self.log(message, "DEBUG")
