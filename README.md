# 🚀 ML Learning Lab

Machine learning development portfolio. Neural networks, computer vision, NLP, and deployment projects.

---

## 📊 Progress Tracker

### Week 1: Neural Network Fundamentals  
**Duration:** October 16-22, 2025 ✅

**Topics Covered:**
- Neural network architecture and forward propagation
- Gradient descent and optimization algorithms
- Backpropagation and automatic differentiation
- PyTorch fundamentals and training loops

**Projects:**
- Built neural network from scratch
- MNIST digit classifier: 91.28% accuracy
- MNIST with CNN: 98.92% accuracy  
- CIFAR-10 with ResNet18: 95.70% accuracy

---

### Week 2: Computer Vision & Object Detection
**Duration:** October 23-29, 2025 ✅

**Topics Covered:**
- Convolutional Neural Networks (CNNs)
- Object detection fundamentals
- YOLO architecture and real-time detection
- Transfer learning with pre-trained models
- Data augmentation techniques
- Model evaluation metrics (mAP, precision, recall)

**Projects:**
- ResNet image classification
- YOLO object detection exploration
- **Safety Equipment Detection System (YOLOv8)**
  - 6-class detection: hard hats, vests, masks, gloves, glasses, boots
  - Performance: 75.1% mAP50-95
  - Real-world construction safety application

---

### Week 3: Medical Image Classification
**Duration:** November 11-17, 2025 ✅

**Topics Covered:**
- Transfer learning with ResNet50
- Medical imaging preprocessing
- Class imbalance handling
- Grad-CAM explainability
- Model interpretability for healthcare

**Projects:**
- **MediScan - Chest X-Ray Pneumonia Classifier**
  - Binary classification: Normal vs Pneumonia
  - Performance: 94.48% validation accuracy
  - Implemented Grad-CAM for visual explanations
  - Deployed on Streamlit Cloud

---

### Weeks 4-6: AI Security & Surveillance System
**Duration:** November 18 - December 8, 2025 ✅

**Topics Covered:**
- Multi-object tracking (DeepSORT)
- Face recognition (FaceNet)
- Real-time video processing
- Alert systems and database integration
- REST API development (FastAPI)
- Dashboard development

**Projects:**
- **Complete AI Security System**
  - Multi-object tracking: 27 FPS real-time performance
  - Face recognition: Known/unknown classification
  - Safety violation detection integration
  - Real-time alerts and analytics dashboard
  - Production deployment with Docker

---

### Week 7: NLP & Sentiment Analysis (In Progress) 🔄
**Duration:** December 9-15, 2025

**Topics Covered:**
- Natural Language Processing fundamentals
- Text preprocessing and tokenization
- Word embeddings and vocabulary building
- RNN/LSTM architecture for sequence modeling
- Hyperparameter optimization strategies
- Bidirectional LSTM and gradient clipping

**Projects:**
- **TextAI Studio - LSTM Sentiment Analyzer**
  - Dataset: IMDB 50K movie reviews
  - Baseline accuracy: 50.61% → Extended: 80.38% (+29.76% improvement)
  - Systematic hyperparameter tuning (learning rates, hidden dims, dropout, layers)
  - Architecture: 2-layer LSTM, 128 hidden units, 2.2M parameters
  - Training: 13 epochs with convergence monitoring
  - Status: Day 45 - Hyperparameter experiments complete

**Progress:**
- ✅ Day 43: NLP fundamentals, IMDB setup, complete preprocessing pipeline
- ✅ Day 44: LSTM model implementation, training (3 epochs), full evaluation
- ✅ Day 45 (Parts 1-2): Extended training (13 epochs), hyperparameter experiments
- 🔄 Day 45 (Parts 3-5): Advanced techniques, model selection, final deployment

---

## 🛠️ Tech Stack

- **Frameworks:** PyTorch 2.0, Ultralytics YOLOv8, Hugging Face Transformers
- **NLP Libraries:** NLTK, Datasets, Tokenizers
- **CV Libraries:** OpenCV, Torchvision
- **Core Libraries:** NumPy, Pandas, Matplotlib, Seaborn
- **Deployment:** Streamlit, FastAPI, Docker
- **Tools:** Jupyter Notebook, Git, Roboflow
- **Language:** Python 3.11

---

## 📁 Repository Structure
```
ml-learning-lab/
├── week1_neural_networks/
│   ├── day01_introduction.ipynb
│   ├── day02_gradient_descent.ipynb
│   ├── day03_backprop_pytorch.ipynb
│   ├── day04_cnns.ipynb
│   ├── day05_advanced_techniques.ipynb
│   ├── day06_cifar10_project.ipynb
│   └── day07_week1_review.ipynb
├── week2_computer_vision/
│   ├── day08_resnet.ipynb
│   ├── day09_object_detection.ipynb
│   ├── day10_YOLO_intro.ipynb
│   ├── day11_YOLO_deepdive.ipynb
│   ├── day12_dataset_theory.ipynb
│   ├── day13_training_theory.ipynb
│   └── [safety detection project files]
├── week3_medical_imaging/
│   └── [MediScan project files]
├── week4-6_security_system/
│   └── [AI Security & Surveillance files]
├── week_7_nlp_fundamentals_sentiment_analysis/
│   ├── day_43_nlp_fundamentals.ipynb
│   ├── day_44_lstm_model_training.ipynb
│   ├── day_45_model_optimization.ipynb
│   ├── data/
│   ├── models/
│   └── results/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📈 Results Summary

| Project | Model Type | Performance | Key Learning |
|---------|-----------|-------------|--------------|
| MNIST Baseline | Fully Connected NN | 91.28% accuracy | Training loops, optimization |
| MNIST CNN | Convolutional Network | 98.92% accuracy | Spatial feature learning |
| CIFAR-10 | ResNet18 Transfer Learning | 95.70% accuracy | Transfer learning efficiency |
| Safety Equipment Detection | YOLOv8 | 75.1% mAP50-95 | Real-time object detection |
| MediScan | ResNet50 + Grad-CAM | 94.48% accuracy | Medical imaging, explainability |
| AI Security System | YOLO + DeepSORT + FaceNet | 27 FPS real-time | Multi-model integration, production deployment |
| **LSTM Sentiment Analyzer** | **2-Layer LSTM** | **80.38% accuracy** | **Sequence modeling, hyperparameter tuning, NLP preprocessing** |

---

## 🎯 Current Focus

**Phase:** Week 7 - Natural Language Processing
- Completing LSTM sentiment analysis optimization
- Implementing advanced techniques (bidirectional LSTM, gradient clipping)
- Building TextAI Studio interface
- Preparing for Week 8: Transformer models and BERT

**Skills Developing:**
- Deep learning for NLP
- Systematic hyperparameter optimization
- Model evaluation and selection
- Production-ready NLP pipelines

---

## 🏆 Key Achievements

- **5 Major Projects Completed** across CV, Medical Imaging, and Security
- **2.2M+ Parameters** trained in LSTM sentiment model
- **29.76% Accuracy Improvement** through systematic optimization
- **Real-time Systems:** 27 FPS tracking, live sentiment analysis
- **Production Deployments:** Streamlit apps, Docker containers, REST APIs

---

## 📚 Learning Journey

**Weeks 1-2:** Neural Networks & Computer Vision Fundamentals  
**Week 3:** Transfer Learning & Medical AI  
**Weeks 4-6:** Production ML Systems & Integration  
**Week 7:** NLP & Sequence Modeling (Current) 🔄  
**Weeks 8-9:** Transformers & Advanced NLP (Upcoming)  
**Weeks 10-12:** Reinforcement Learning (Planned)

---

## 🔗 Related Projects

- [Safety Equipment Detection System](https://github.com/01-Audrey/safety-equipment-detector) - YOLOv8-based real-time PPE detection (75.1% mAP)
- [MediScan](https://github.com/01-Audrey/mediscan) - Chest X-ray pneumonia classifier with Grad-CAM (94.48% accuracy)
- [AI Security & Surveillance System](https://github.com/01-Audrey/ai-security-surveillance-system) - Complete security solution with tracking, face recognition, and analytics (27 FPS)
---

**Last Updated:** December 12, 2025  
**Current Day:** Day 45 of 168 (26.8% complete)  
**Next Milestone:** Complete TextAI Studio (Week 7)
