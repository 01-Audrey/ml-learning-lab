
# 🔒 AI Security & Surveillance System

> An intelligent security and safety monitoring system combining face recognition, person tracking, PPE detection, and behavioral analysis with real-time alerts.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![Status](https://img.shields.io/badge/Status-Week%205%20Complete-success.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Performance](#performance)
- [Installation](#installation)
- [Usage](#usage)
- [Week 5 Progress](#week-5-progress)
- [Future Work](#future-work)
- [Tech Stack](#tech-stack)

---

## 🎯 Overview

This project implements a comprehensive security and safety monitoring system developed over **Week 5 (Days 29-35)** of my ML learning journey. The system provides real-time face recognition, person tracking, PPE compliance monitoring, and intelligent alerting for facility security applications.

**Use Cases:**
- Office buildings (visitor management, access control)
- Construction sites (PPE compliance, safety zones)
- Warehouses (safety monitoring, restricted areas)
- Manufacturing facilities (compliance tracking)

---

## ✨ Features

### 🔐 Security Features
- **Face Detection**: Real-time face detection using DNN Caffe model (ResNet-10)
- **Face Recognition**: 512-dimensional embeddings with ResNet18
- **Person Tracking**: Centroid tracker with occlusion handling (27 FPS)
- **Unknown Person Detection**: Automatic identification of unknown individuals
- **Multi-Camera Support**: Ready for multiple camera streams

### 🦺 Safety Features
- **PPE Detection**: 6 categories (hardhat, vest, mask, gloves, goggles, boots)
- **Safety Zones**: Polygon-based restricted area monitoring
- **Behavior Analysis**: Running, loitering, dwell time tracking
- **Compliance Tracking**: Real-time safety compliance metrics

### 🚨 Alert System
- **Smart Alerts**: Quality gating, deduplication, cooldown management
- **Multi-Channel Notifications**: Email, SMS, webhooks
- **Escalation Logic**: Time-based and repeat offender escalation
- **Alert Optimization**: 73% false positive reduction (15% → 4%)

### 📊 Analytics & Reporting
- **Real-Time Dashboard**: Live monitoring with statistics
- **Compliance Reports**: Daily/weekly safety reports
- **Performance Metrics**: FPS, accuracy, response times
- **Alert History**: Searchable database with audit trail

---

## 🏗️ System Architecture
```
Input → Face Detection → Tracking → Recognition → Safety Analysis → Alerts → Output
         (Day 29)       (Day 31)    (Day 30)       (Day 34)      (Day 32-33)
```

**Pipeline Components:**
1. **Detection Layer**: DNN face detector (ResNet-10)
2. **Tracking Layer**: Centroid tracker with ID persistence
3. **Recognition Layer**: ResNet18 embeddings + cosine similarity
4. **Safety Layer**: PPE detection, zones, behavior analysis
5. **Alert Layer**: Rules engine, notifications, escalation
6. **Storage Layer**: SQLite databases for alerts and faces

---

## ⚡ Performance

| Metric | Value |
|--------|-------|
| **Processing Speed** | 27 FPS (real-time) |
| **Detection Latency** | ~15ms per frame |
| **Recognition Latency** | ~5ms per face |
| **False Positive Rate** | 4% (after optimization) |
| **False Positive Reduction** | 73% improvement |
| **Compliance Tracking** | Real-time |

**Optimizations:**
- ✅ Recognition caching (every 10 frames) - 10x speedup
- ✅ Quality gating (confidence, blur, size filters)
- ✅ Unknown person deduplication
- ✅ Alert cooldown management

---

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
pip
```

### Clone Repository
```bash
git clone https://github.com/yourusername/ml-learning-lab.git
cd ml-learning-lab/week5_face_recognition
```

### Install Dependencies
```bash
pip install opencv-python pillow matplotlib seaborn ultralytics
pip install numpy pandas torch torchvision scipy scikit-learn
```

### Download Models
```bash
# Face detection model (DNN Caffe)
# Place in models/ directory:
# - deploy.prototxt
# - res10_300x300_ssd_iter_140000.caffemodel

# Eye detection cascade
# - haarcascade_eye.xml
```

---

## 💻 Usage

### Basic Face Recognition
```python
from integrated_pipeline import IntegratedSecurityPipeline

# Initialize pipeline
pipeline = IntegratedSecurityPipeline(face_net, matcher)

# Process frame
annotated_frame, results = pipeline.process_frame(frame)

# Get statistics
stats = pipeline.get_statistics()
print(f"FPS: {stats['avg_fps']:.1f}")
```

### Webcam Demo
```python
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated, results = pipeline.process_frame(frame)
    cv2.imshow('Security System', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Alert Configuration
```json
{
  "unknown_person": {
    "enabled": true,
    "priority": "critical",
    "cooldown_seconds": 30
  },
  "after_hours": {
    "enabled": true,
    "priority": "high",
    "time_range": ["22:00", "06:00"]
  }
}
```

---

## 📅 Week 5 Progress

| Day | Topic | Status |
|-----|-------|--------|
| **Day 29** | Face Detection Foundation | ✅ Complete |
| **Day 30** | Face Embeddings & Matching | ✅ Complete |
| **Day 31** | Video Tracking Integration | ✅ Complete |
| **Day 32** | Alert System Implementation | ✅ Complete |
| **Day 33** | Alert Optimization | ✅ Complete |
| **Day 34** | Safety Violation Detection | ✅ Complete |
| **Day 35** | Integration Testing | ✅ Complete |

**Week 5: 100% Complete! 🎉**

---

## 🔮 Future Work (Week 6)

### Planned Features
- 🌐 REST API (FastAPI)
- 📊 Web Dashboard (Streamlit/React)
- 🐳 Docker Containerization
- 📈 Advanced Analytics (heat maps, traffic patterns)
- 🔄 Multi-Camera Synchronization
- ☁️ Cloud Deployment

---

## 🛠️ Tech Stack

**Core Technologies:**
- **Python 3.8+**: Primary language
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision
- **NumPy/SciPy**: Numerical computing

**Models:**
- **Face Detection**: DNN Caffe (ResNet-10)
- **Face Recognition**: ResNet18 (512-dim embeddings)
- **PPE Detection**: YOLOv8

**Databases:**
- **SQLite**: Alert history, face database
- **JSON**: Configuration, metadata

**Notifications:**
- **SMTP**: Email alerts
- **Twilio API**: SMS alerts (optional)
- **Webhooks**: HTTP POST integration

---

## 📊 Project Structure
```
week5_face_recognition/
├── day29_face_detection.ipynb
├── day30_face_embeddings.ipynb
├── day31_video_tracking.ipynb
├── day32_alert_system.ipynb
├── day33_alert_optimization.ipynb
├── day34_safety_violations.ipynb
├── day35_week5_integration_testing.ipynb
├── models/
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   └── haarcascade_eye.xml
├── face_database/
│   ├── database.json
│   └── known_faces/
├── alerts.db
├── alert_rules.json
└── results/
```

---

## 📝 License

This project is part of my personal ML learning journey.

---

## 👤 Author

**Audrey**
- 🎓 3rd Year CS Student (LPU Laguna)
- 🎯 Specialization: Game Development + AI
- 🚀 Goal: ML/AI Engineer Internship (Summer 2026)
- 📍 Location: Philippines

---

## 🙏 Acknowledgments

- Week 5 of 24-week ML Engineer learning path
- Part of Major Project #1: AI Security & Surveillance System
- Days 29-35: Face Recognition & Alerts complete
- Next: Week 6 - Dashboard, API & Deployment

---

**⭐ If you find this project helpful, please star the repository!**

---

*Generated: 2025-11-28*
*Week 5 Complete | Days 29-35*
