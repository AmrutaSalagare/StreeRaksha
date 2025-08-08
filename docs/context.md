# StreeRaksha - AI & IoT-Powered Women Safety Analytics

## 1. Introduction

**StreeRaksha** is an advanced AI and IoT-powered platform dedicated to enhancing women’s safety in public spaces. By leveraging real-time computer vision, machine learning, and edge IoT devices (such as Raspberry Pi), StreeRaksha analyzes live CCTV feeds and sensor data to detect potential threats and instantly alert authorities through a secure dashboard. This proactive system aims to prevent incidents before escalation, enabling rapid response and data-driven decision-making for law enforcement and security agencies.

### Problem Statement

Women often face safety challenges in public areas, particularly in high-risk zones or during late hours. Conventional surveillance systems are largely reactive, providing evidence only after incidents occur. StreeRaksha bridges this gap by delivering real-time, intelligent threat detection and immediate notifications to authorities, empowering them to intervene swiftly and effectively.

---

## 2. Core Features

StreeRaksha’s architecture is built around four core modules, each designed to address a critical aspect of public safety:

### 2.1 Real-Time Threat Detection (AI + IoT)

- **AI-Powered Video Analysis**: Continuously monitors live CCTV and edge camera feeds using state-of-the-art computer vision models.
- **IoT Edge Devices**: Raspberry Pi 5 nodes process video and sensor data locally, ensuring low-latency detection and privacy preservation.
- **Detection Parameters**:
  - Gender classification to identify women in the scene.
  - Crowd density analysis to detect potentially unsafe situations (e.g., a lone woman surrounded by a group).
  - Recognition of distress gestures and abnormal movements (e.g., raised hands, frantic motions).
  - Identification of women in isolated or high-risk areas, especially during nighttime.
  - Environmental context from sensors (lighting, sound, motion).
- **Parallel Processing**: Enables rapid, simultaneous analysis of multiple video streams for real-time response.

### 2.2 Authority Dashboard & Incident Management

- **Real-Time Alerts**: Instantly notifies authorities of detected threats with supporting evidence (images, sensor data).
- **Live Feed Monitoring**: Secure access to live CCTV and edge device footage for real-time assessment.
- **Incident Logging & Analytics**: Maintains a comprehensive record of alerts, detections, and responses for review and pattern analysis.
- **Interactive Map Visualization**: Displays incident locations and risk zones to assist in resource allocation and rapid deployment.
- **Incident Response Workflow**:
  - Visualizes alerts on an interactive map.
  - Notifies nearby law enforcement for immediate intervention.
  - Logs actions taken and outcomes for accountability and future strategy.

### 2.3 Intelligent Alert System & Evidence Collection

- **Automated Alerts**: Generates and dispatches real-time notifications to security personnel and authorities upon threat detection.
- **Evidence Storage**: Captures and securely stores image frames and sensor data as verifiable proof for investigations and legal proceedings.
- **Multi-Channel Communication**: Supports notifications via dashboard, SMS, and other secure channels for redundancy and reliability.

### 2.4 Predictive Safety Analytics (For Authorities)

- **Risk Heatmaps**: Aggregates historical incident data to identify and visualize high-risk zones and time periods.
- **Trend Analysis**: Provides insights into recurring patterns, enabling authorities to optimize patrols and preventive measures.
- **Resource Planning**: Assists in strategic deployment of security resources based on data-driven risk assessments.

---

## 3. IoT-Based Approach: Smart Edge Surveillance Nodes

StreeRaksha deploys Raspberry Pi 5 devices as **smart surveillance nodes** at strategic public locations. Each node operates as follows:

- **Live Video & Sensor Capture**: Continuously captures video and environmental sensor data (lighting, sound, motion).
- **On-Device AI Analysis**: Runs lightweight, optimized models locally for immediate threat detection.
- **Edge Processing & Privacy**: Processes sensitive data on-device, transmitting only relevant alerts and evidence to the central platform.
- **Real-Time Threat Detection**: Sends alerts with supporting evidence to the authority dashboard upon detection.
- **Centralized Monitoring & Response**: Integrates with the main platform for coordinated, city-wide monitoring and rapid response.

---

## 4. Tech Stack

StreeRaksha is built using robust, open-source, and free-tier technologies to ensure scalability, maintainability, and cost-effectiveness:

### 4.1 Backend Development

- **Programming Language**: Python (FastAPI)
- **Database**: Supabase (PostgreSQL with PostGIS for geospatial data)
- **Real-Time Messaging**: Redis (Pub/Sub), Supabase Realtime

### 4.2 AI & Computer Vision

- **Video Processing**: OpenCV
- **Object Detection**: YOLO (You Only Look Once)
- **Gesture & Pose Recognition**: MediaPipe
- **Gender Classification**: MobileNetV2

### 4.3 IoT & Edge

- **Edge Devices**: Raspberry Pi 5 (Python, OpenCV, TensorFlow Lite)
- **Communication**: MQTT, HTTP REST APIs, WebSocket

### 4.4 Frontend Development (Authority Dashboard)

- **Web Dashboard**: Dash (Python), Here Maps API, Dash's built-in charts

### 4.5 Communication & Alerts

- **Real-Time Notifications**: Firebase Cloud Messaging (FCM)
- **SMS/Call Alerts**: Twilio (Free Tier)

### 4.6 Cloud & Infrastructure

- **Hosting & Deployment**: Vercel (Frontend), Render (Backend)
- **Database Hosting**: Supabase (Free Tier)
- **Media Storage**: Supabase Storage

### 4.7 DevOps & CI/CD

- **Version Control**: Git, GitHub
- **CI/CD Pipelines**: GitHub Actions

---

## 5. Operational Workflow

The StreeRaksha platform follows a streamlined workflow to ensure efficient threat detection and rapid response:

1. **Camera & IoT Node Onboarding**: Integrate live CCTV and Raspberry Pi edge feeds for real-time monitoring.
2. **AI & Sensor Analysis**: Analyze video frames and sensor data using advanced computer vision and environmental models.
3. **Threat Detection**: Identify potential threats based on predefined safety parameters.
4. **Alert Generation**: Trigger alerts via Redis Pub/Sub or MQTT and send them to the authority dashboard.
5. **Evidence Collection**: Capture and store relevant image frames and sensor data in Supabase for verification and investigation.
6. **Dashboard Notification**: Authorities receive real-time alerts, access live feeds, and review evidence through the secure dashboard.
7. **Incident Response**: Authorities take immediate action, with all incidents and responses logged for future analysis.
8. **Post-Incident Analysis**: Use historical data to identify patterns, improve detection algorithms, and inform preventive strategies.

---

## 6. Benefits & Impact

- **Proactive Safety**: Enables real-time detection and rapid response, preventing incidents before escalation.
- **Scalable & Cost-Effective**: IoT edge nodes and open-source stack support city-wide deployment at minimal cost.
- **Privacy-Preserving**: Edge processing ensures sensitive data remains local, transmitting only essential information.
- **Empowering Authorities**: Provides actionable insights, real-time alerts, and evidence to law enforcement and security teams.
- **Data-Driven Insights**: Facilitates optimization of patrols, resource allocation, and infrastructure improvements based on analytics.

---

## 7. Future Enhancements

- **Mobile App for Public Users**: Planned React Native app for user SOS, incident reporting, safety map, and personalized safe route suggestions.
- **User-Reported Incidents**: Direct reporting and engagement via the mobile app.
- **User Location History & Analytics**: Personalized safety analytics and recommendations for users.
- **Integration with City Emergency Services**: Automated escalation to emergency responders.
- **Advanced AI Models**: Continuous improvement of detection algorithms for higher accuracy and broader threat coverage.

---

## 8. Conclusion

StreeRaksha leverages cutting-edge AI, IoT, and real-time analytics to proactively address women’s safety in public spaces. By integrating intelligent surveillance, edge computing, predictive analytics, and rapid response mechanisms, the platform ensures that potential threats are detected and mitigated before escalation. With its scalable, open-source tech stack and streamlined workflow, StreeRaksha empowers authorities to create safer environments and lays the foundation for future public engagement
