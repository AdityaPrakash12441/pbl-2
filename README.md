Wildlife Guardian 🐅

AI-Powered Wildlife Monitoring & Poacher Detection System

An intelligent real-time monitoring system that uses computer vision to detect endangered species, classify wildlife, and identify potential poachers or human intruders in protected wildlife areas.

🚀 Live Demo

Access the hosted application here: [website->](wildlifeguardian.streamlit.app)

How to Use the Web Application

*Access the Dashboard
   - Open the link above in your web browser
   - The dashboard will load with a welcome screen

*Start Monitoring
   - Click the "Start Monitoring" button on the main page
   - Your browser will request permission to access your webcam
   - Grant permission to begin live detection

*View Real-Time Detections
   - The live video feed appears in the center with annotated detections
   - Green boxes = Wildlife animals detected
   - Red boxes = Poacher/Human threat detected
   - Purple boxes = Endangered species detected

*Monitor System Metrics
   - Detections (current frame): Number of animals in this frame
   - Threats (current frame): Number of humans/poachers detected
   - Threat log entries: Total intrusions logged
   - Species Frequency: Bar chart of top 10 species detected

*Navigate Different Views
   - Live Monitoring: Real-time video feed with metrics
   - Threat Log (Poachers Only): All human intrusions detected with timestamps
   - Endangered Species Log: All endangered species sightings
   - Wildlife Log: General wildlife detections

*Manage Logs
   - Click "Stop Monitoring" to pause detection
   - Click "Clear Logs" to reset all data and start fresh
   - All logs are stored in the session and clear when you refresh

*System Information
   - Device shows if GPU (CUDA) or CPU is being used
   - YOLO model and Classifier model versions displayed

📋 Project Overview

What is Wildlife Guardian?

Wildlife Guardian is an intelligent computer vision system designed to monitor protected wildlife areas in real-time. It combines:

YOLO v8 Object Detection for detecting animals and humans

ResNet50 Classification for species identification

Real-time Analytics for threat assessment and logging

Key Features

✅ Real-Time Detection

Processes video frames in real-time (30 FPS)

Detects 10+ different wildlife species

Identifies human threats (poachers/intruders)

✅ Endangered Species Tracking

Automatically flags endangered species sightings

Maintains historical logs with timestamps

Calculates intrusion duration metrics

✅ Threat Detection

Distinguishes poachers/humans from animals

Tracks intrusion count and duration

Alerts on detected threats

✅ Species Classification

Classifies detected animals into specific species

Uses ImageNet classification with fine-tuned thresholds

Maintains species frequency statistics

✅ Analytics Dashboard

Live species frequency charts

Detection/threat metrics

Historical threat logs

Session-based statistics

🔧 Local Setup (For Developers)

Prerequisites

Python 3.8+

Webcam (for local testing)

~2-4GB RAM minimum

Performs well on CPU too

Installation

Clone the repository
   bash    git clone https://github.com/AdityaPrakash12441/pbl-2.git    cd pbl-2
