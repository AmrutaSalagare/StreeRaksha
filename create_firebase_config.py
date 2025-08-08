#!/usr/bin/env python3
"""
StreeRaksha - Firebase Configuration Generator

This script generates a firebase_config.json file for the StreeRaksha application.
"""

import json
import os
import sys
import getpass


def create_firebase_config():
    """Generate a Firebase configuration file"""
    print("\n===== StreeRaksha Firebase Configuration Generator =====\n")
    print("This tool will help you create a firebase_config.json file for Firebase integration.")
    print("You'll need your Firebase project information from the Firebase console.")
    print("Visit: https://console.firebase.google.com/\n")

    config = {}

    # Get Firebase project information
    config['apiKey'] = input("Enter your Firebase API Key: ")
    config['authDomain'] = input(
        "Enter your Firebase Auth Domain (e.g., project-id.firebaseapp.com): ")
    config['projectId'] = input("Enter your Firebase Project ID: ")
    config['storageBucket'] = input(
        "Enter your Firebase Storage Bucket (e.g., project-id.appspot.com): ")
    config['databaseURL'] = input(
        "Enter your Firebase Database URL (e.g., https://project-id.firebaseio.com): ")
    config['messagingSenderId'] = input(
        "Enter your Firebase Messaging Sender ID: ")
    config['appId'] = input("Enter your Firebase App ID: ")

    # Optional service account credentials for server-side authentication
    use_service_account = input(
        "\nDo you want to configure service account credentials? (y/n): ").lower()
    if use_service_account == 'y':
        config['serviceAccount'] = {
            'type': "service_account",
            'client_email': input("Enter service account client email: "),
            'private_key': input("Enter service account private key (paste the entire key): "),
            'client_id': input("Enter service account client ID: ")
        }

    # Write the configuration to a file
    config_path = os.path.join(os.path.dirname(
        __file__), 'firebase_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nFirebase configuration saved to: {config_path}")
    print("\nIMPORTANT: Make sure to keep this file secure and DO NOT commit it to version control!")
    print("You may want to add firebase_config.json to your .gitignore file.\n")


def create_sample_config():
    """Create a sample config file for testing"""
    sample_config = {
        'apiKey': "sample-api-key-for-testing",
        'authDomain': "streeraksha-project.firebaseapp.com",
        'projectId': "streeraksha-project",
        'storageBucket': "streeraksha-project.appspot.com",
        'databaseURL': "https://streeraksha-project.firebaseio.com",
        'messagingSenderId': "123456789012",
        'appId': "1:123456789012:web:abc123def456"
    }

    # Write the sample configuration to a file
    config_path = os.path.join(os.path.dirname(
        __file__), 'firebase_config.json')
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)

    print(f"\nSample Firebase configuration saved to: {config_path}")
    print("This configuration is for TESTING ONLY and will not connect to a real Firebase project.")
    print("Replace with actual credentials when ready for production.\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--sample":
        create_sample_config()
    else:
        create_firebase_config()
