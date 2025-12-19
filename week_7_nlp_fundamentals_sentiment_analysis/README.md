# 🎭 TextAI Studio - LSTM Sentiment Analyzer

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**A professional LSTM-powered sentiment analysis web application built with PyTorch and Streamlit.**

Analyze movie reviews and discover the sentiment behind the words with real-time predictions, word importance visualization, and batch processing capabilities.

---

## 🌟 Features

### 🔍 Single Text Analysis
- Real-time sentiment prediction
- Confidence scores and detailed metrics
- Word importance visualization
- 5 pre-loaded example reviews
- Natural language explanations

### 📊 Batch Processing
- Analyze multiple reviews at once
- Progress tracking
- Summary statistics
- Export results (CSV/JSON)

### 🎨 Professional Interface
- Clean, intuitive design
- Color-coded sentiment display
- Interactive visualizations
- Responsive layout
- Mobile-friendly

---

## 🚀 Live Demo

**Try it now:** [TextAI Studio on Streamlit Cloud](#) *(Link will be added after deployment)*

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 80.38% |
| **Precision** | 80.44% |
| **Recall** | 80.36% |
| **F1 Score** | 80.40% |
| **Inference Speed** | <50ms |

---

## 🧠 Technical Details

### Architecture
- **Model:** Bidirectional LSTM (2 layers, 128 hidden units)
- **Vocabulary:** 20,000 unique words
- **Training Data:** IMDB 50,000 movie reviews
- **Parameters:** 2.2M+
- **Framework:** PyTorch 2.0.1

### Training Journey
- **Duration:** 5 days (Days 43-47)
- **Initial Accuracy:** 50.61% (Day 44, 3 epochs)
- **Final Accuracy:** 80.38% (Day 45, 13 epochs)
- **Optimization:** Extensive hyperparameter tuning
- **Validation:** Comprehensive robustness testing

---

## 🛠️ Technology Stack

- **Deep Learning:** PyTorch
- **Web Framework:** Streamlit
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Language:** Python 3.11+

---

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- Git

### Local Setup

1. **Clone the repository**
```bash
   git clone https://github.com/01-Audrey/ml-learning-lab.git
   cd ml-learning-lab/week_7_nlp_fundamentals_sentiment_analysis
```

2. **Install dependencies**
```bash
   pip install -r requirements.txt
```

3. **Run the application**
```bash
   streamlit run app.py
```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`

---

## 📖 Usage

### Single Text Analysis
1. Select **"🔍 Single Text Analysis"** mode
2. Choose to type your own text or use example reviews
3. Click **"🚀 Analyze Sentiment"**
4. View results with word importance visualization

### Batch Processing
1. Select **"📊 Batch Processing"** mode
2. Enter multiple reviews (one per line)
3. Click **"🚀 Analyze All Reviews"**
4. View summary statistics and detailed results

---

## 📁 Project Structure
```
week_7_nlp_fundamentals_sentiment_analysis/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
│
├── data/
│   └── preprocessing_objects.pkl   # Vocabulary and preprocessing
│
├── models/
│   ├── best_model_extended_v2.pt   # Trained LSTM weights
│   └── experiments/                # Training experiments
│       └── day45_all_experiments.pkl
│
├── results/                        # Evaluation results
│   ├── day_46/                     # Final evaluation
│   ├── day_47/                     # Robustness testing
│   └── day_48_summary.txt          # Interface summary
│
└── notebooks/                      # Development notebooks
    ├── day_43_nlp_fundamentals.ipynb
    ├── day_44_lstm_training.ipynb
    ├── day_45_optimization.ipynb
    ├── day_46_final_evaluation.ipynb
    ├── day_47_robustness_explainability.ipynb
    └── day_48_streamlit_interface.ipynb
```

---

## 🎯 Development Journey

### Week 7: NLP Fundamentals & Sentiment Analysis (5 days)

**Day 43:** NLP Fundamentals & Data Preprocessing
- Text cleaning and tokenization
- Vocabulary building (20K words)
- Sequence padding and batching
- IMDB dataset exploration

**Day 44:** LSTM Model Architecture & Initial Training
- Built bidirectional LSTM from scratch
- Initial training (3 epochs)
- Achieved 50.61% validation accuracy
- Established baseline performance

**Day 45:** Hyperparameter Optimization
- Extended training to 13 epochs
- Tested multiple configurations
- Optimized learning rate, hidden dims, dropout
- **Achieved 80.38% validation accuracy**

**Day 46:** Comprehensive Evaluation & Testing
- Full test set evaluation (25K samples)
- Error analysis and pattern identification
- Calibration curve analysis
- Performance profiling

**Day 47:** Robustness & Explainability
- Adversarial robustness testing
- Cross-domain evaluation
- Word importance visualization
- Production readiness assessment

**Day 48:** Streamlit Interface Development
- Built complete web application
- Real-time prediction interface
- Batch processing functionality
- Professional UI/UX design

---

## 🏆 Key Achievements

✅ **End-to-end ML project** from data preprocessing to deployment  
✅ **80%+ accuracy** on IMDB sentiment classification  
✅ **Production-ready** with comprehensive testing  
✅ **Robust** to typos, noise, and cross-domain data  
✅ **Explainable** with word importance visualization  
✅ **Professional** web interface with Streamlit  
✅ **Well-documented** with detailed notebooks  

---

## 📊 Model Capabilities

### Strengths
- Strong performance on movie reviews (80.38%)
- Fast inference (<50ms per prediction)
- Robust to typos and text variations
- Good cross-domain generalization
- Interpretable predictions

### Limitations
- Trained only on English text
- Optimized for movie reviews
- May struggle with heavy sarcasm
- Best performance on 50-200 word reviews

---

## 🔮 Future Improvements

- [ ] Fine-tune on domain-specific data
- [ ] Add multilingual support
- [ ] Implement attention mechanism
- [ ] Support for aspect-based sentiment
- [ ] Integration with review platforms
- [ ] Mobile app development

---

## 👤 Author

**Audrey**
- Computer Science Student 
- [GitHub](https://github.com/01-Audrey)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Training Data:** IMDB Movie Review Dataset
- **Framework:** PyTorch team for excellent deep learning tools
- **Interface:** Streamlit for making ML deployment accessible
- **Inspiration:** Part of 24-week ML learning journey

---

## 📚 Citation

If you use this project in your research or work, please cite:
```bibtex
@software{textai_studio,
  author = {Audrey},
  title = {TextAI Studio: LSTM Sentiment Analyzer},
  year = {2025},
  url = {https://github.com/01-Audrey/ml-learning-lab}
}
```

---

## 💬 Contact

Have questions or suggestions? Feel free to:
- Open an issue on GitHub
- Connect via GitHub profile

---

**Built with ❤️ using PyTorch and Streamlit**

*Part of a 24-week ML learning journey (Week 7 of 24)*
