# StreeRaksha: AI & IoT-Powered Women Safety Analytics

## Overview

StreeRaksha is an AI and IoT-powered platform designed to enhance public safety for women through real-time threat detection and rapid response mechanisms. By leveraging advanced computer vision, machine learning, and edge IoT devices (like Raspberry Pi), the system analyzes live CCTV feeds and sensor data to identify potential threats and alerts authorities via an integrated dashboard.

## Problem Statement

Women face unique safety challenges in public spaces, especially in high-risk areas or during late hours. Traditional surveillance systems are reactive and often fail to prevent incidents. StreeRaksha addresses this gap by using AI and IoT to detect threats in real-time and notify authorities instantly, enabling quicker interventions and empowering women with actionable safety information.

## Project Structure

The project is organized into modular components for better debugging and understanding:

```
StreeRaksha/
├── main.py               # Main application entry point
├── detector.py           # Core detector module (contains StreeRakshaDetector class)
├── pose_analyzer.py      # Pose analysis functionality
├── gender_detector.py    # Gender detection module
├── tracker.py            # Person tracking module with periodic gender reclassification
├── thread_manager.py     # Advanced multi-threading functionality with priority queues and adaptive processing
├── thread_health_monitor.py # Thread health monitoring and performance analytics
├── priority_queue.py    # Priority-based queues for efficient inter-thread communication
├── config_manager.py    # Dynamic configuration management for threads
├── firebase_manager.py   # Firebase integration for backend services
├── alert_system.py       # Alert generation and risk assessment
├── visualizer.py         # Visualization and UI rendering
├── logger.py             # Logging functionality
├── debugger.py           # Debug utilities
├── reset_camera.py       # Camera reset utility
├── firebase_config.json  # Firebase configuration (with .example template)
├── requirements.txt      # Project dependencies
├── yolov8n.pt            # YOLOv8 model file
├── evidence/             # Saved evidence images and sensor data
├── logs/                 # Application logs
├── docs/                 # Documentation
└── __pycache__/          # Python bytecode cache
```

## Module Descriptions

- **main.py**: Entry point for the application that coordinates all components
- **detector.py**: Main detector module that processes video frames and coordinates detection pipeline
- **pose_analyzer.py**: Analyzes human poses to detect distress signals and vulnerable positions
- **gender_detector.py**: Handles gender classification using neural networks or simulation
- **tracker.py**: Tracks persons across video frames to maintain identity consistency with periodic gender reclassification
- **alert_system.py**: Risk assessment and alert generation system
- **visualizer.py**: Handles UI rendering and visualization
- **logger.py**: Logging system for debugging and audit trails
- **debugger.py**: Debug utilities for development and testing
- **reset_camera.py**: Utility to reset camera connections when issues occur

## Core Features

### 1. Periodic Gender Reclassification

The system implements an advanced gender reclassification mechanism to improve accuracy over time:

- Automatically re-evaluates gender classification every 3 seconds
- Refreshes more frequently for low confidence detections
- Applies hysteresis to prevent gender flip-flopping
- Maintains gender classification history for improved stability
- Requires higher confidence thresholds to change established gender classifications

### 2. Real-Time Threat Detection (AI + IoT)

- AI-powered video analysis using computer vision models
- Edge processing via Raspberry Pi 5 nodes
- Detection of gender, crowd density, distress gestures, and environmental context
- Low-latency parallel processing for immediate response

### 3. Predictive Safety Analytics

- Real-time safety ratings with color-coded zones (Red/Yellow/Green)
- Safe route suggestions for high-risk areas
- Personal dashboard for tracking safety trends

### 4. Intelligent Alert System & Evidence Collection

- Automated real-time alerts for security personnel
- User-reported incident tracking
- Evidence capture and storage for verification

### 5. Advanced Multi-Threading Architecture

- Optimized frame capture, processing, and rendering threads
- Priority-based queuing with motion detection prioritization
- Adaptive processing for consistent performance under load
- Thread health monitoring and performance analytics
- Dynamic configuration management for thread parameters
- Efficient resource utilization with thread priority management
- Real-time performance metrics and bottleneck detection

### 6. Actionable Insights Dashboard

- Real-time alerts for authorities
- Live feed monitoring
- Incident logging and response workflow
- Interactive map visualization

### 7. Mobile App Features

- Quick SOS button with location sharing
- Customizable alert settings
- User location history analysis
- Admin dashboard with safety heatmaps

## Tech Stack

### Backend

- Python (FastAPI)
- Supabase (PostgreSQL with PostGIS)
- Redis, Supabase Realtime

### AI & Computer Vision

- OpenCV
- YOLO (You Only Look Once)
- MediaPipe
- MobileNetV2

### Multi-Threading & Performance

- Priority-based task scheduling
- Advanced thread synchronization with queues
- Real-time thread health monitoring
- Adaptive processing for resource optimization
- Performance metrics collection and analysis

### IoT & Edge

- Raspberry Pi 5 with Python, OpenCV, TensorFlow Lite
- MQTT, REST APIs, WebSocket

### Frontend

- Web Dashboard: Dash (Python), Here Maps API
- Mobile App: React Native, OpenStreetMap with Leaflet.js

### Communication & Infrastructure

- Firebase Cloud Messaging
- Twilio
- Vercel, Render, Supabase
- Git, GitHub Actions

## Getting Started

### Prerequisites

- Python 3.8+
- Raspberry Pi 5 (for edge nodes)
- CCTV or webcam access

### Installation

```bash
# Clone the repository
git clone https://github.com/HemanthKumar-CS/StreeRaksha.git
cd StreeRaksha

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables (if needed)
```

### Running the Application

```bash
# Run the main application
python main.py

# To debug or reset the camera if issues occur
python reset_camera.py
```

### Development

For development with enhanced logging and debug visualizations:

```python
# In main.py
debugger.debug_overlay = True  # Enable debug overlay
```

## Project Structure

```
StreeRaksha/
├── app.py               # Main application entry point
├── api/                 # Backend API endpoints
├── models/              # AI models and detection logic
├── edge/                # Edge device code for Raspberry Pi
├── dashboard/           # Web dashboard
├── mobile/              # Mobile app code
└── docs/                # Documentation
```

## Benefits & Impact

- Proactive safety through real-time detection and rapid response
- Scalable, cost-effective implementation using edge nodes
- Privacy-preserving design with edge processing
- Empowering users with actionable safety information
- Data-driven insights for optimizing security resources

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Legal Information

StreeRaksha is proprietary software. All rights reserved. © 2024 StreeRaksha Team.

Unauthorized copying, modification, distribution, or use of this software is strictly prohibited. This repository is shared for demonstration purposes only.

## Contributors

- Hemanth Kumar C S
- Vandana H
- Amruta Salagare
- Reem K
