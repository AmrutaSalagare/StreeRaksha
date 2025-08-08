# StreeRaksha Project TODO

This file tracks the core development tasks for the StreeRaksha platform. All solutions should use the best free and open-source platforms and services available.

---

## Completed Tasks

- [x] Implement periodic gender re-classification to improve detection accuracy and stability over time.
- [x] Add hysteresis to gender detection to prevent frequent changes in classification.
- [x] Improve tracking to maintain consistent person identities across frames.
- [x] Implement advanced multi-threading architecture with priority queues for optimal performance.
- [x] Create thread health monitoring system for performance analysis and bottleneck detection.
- [x] Implement adaptive processing to maintain consistent FPS under varying loads.
- [x] Add dynamic configuration management for thread parameters and system settings.

## 1. Edge Device (Raspberry Pi 5) Integration

- [ ] Port and optimize detection code to run efficiently on Raspberry Pi 5.
- [ ] Integrate environmental sensors (lighting, sound, motion) with the Pi and process their data.
- [ ] Test real-time video capture, inference, and alert generation on the Pi.
- [ ] Ensure only relevant alerts and evidence are sent to the backend (not full video streams).

## 2. Backend & API (FastAPI)

- [ ] Implement FastAPI backend to receive alerts and evidence from edge devices.
- [ ] Create endpoints for dashboard data (alerts, incidents, analytics).
- [ ] Integrate with Supabase (PostgreSQL) for incident logs, evidence, and geospatial data.
- [ ] Implement authentication and secure data transfer.

## 3. Real-Time Communication

- [ ] Integrate Redis Pub/Sub or MQTT for real-time alert delivery from edge devices to backend/dashboard.
- [ ] Implement notification system using free tiers (e.g., FCM for push, Twilio for SMS/calls).

## 4. Authority Dashboard (Dash)

- [ ] Develop dashboard UI using Dash (Python) and Here Maps API.
- [ ] Display real-time alerts and incident locations on an interactive map.
- [ ] Enable live feed monitoring from edge devices.
- [ ] Provide incident review, analytics, and response logging.

## 5. Evidence Management

- [ ] Automate upload of image frames and sensor data to Supabase Storage.
- [ ] Link uploaded evidence to incident records in the database.
- [ ] Ensure secure and organized storage of all evidence.

## 6. Predictive Analytics

- [ ] Implement risk heatmaps using historical incident data.
- [ ] Add trend analysis and reporting tools for authorities.
- [ ] Develop resource planning features based on analytics.

## 7. DevOps & Documentation

- [ ] Add deployment scripts for backend, dashboard, and edge devices (using free platforms like Render, Vercel).
- [ ] Set up CI/CD pipelines with GitHub Actions.
- [ ] Expand technical and deployment documentation for contributors and users.
