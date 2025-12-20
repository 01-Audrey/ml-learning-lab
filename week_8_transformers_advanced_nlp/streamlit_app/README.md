# Job-Resume Matcher - Streamlit App

## 🚀 Quick Start

### Installation
```bash
pip install streamlit sentence-transformers PyPDF2 python-docx plotly
```

### Running the App
```bash
cd streamlit_app
streamlit run job_matcher_app.py
```

The app will open at: http://localhost:8501

## 📋 Features

### Single Resume Mode
1. Upload a resume (PDF, DOCX, or TXT)
2. View detected skills
3. Click "Find Matching Jobs"
4. See top job matches with scores
5. Download match report

### Batch Processing Mode
1. Upload multiple resumes
2. Click "Process All Resumes"
3. View batch results table
4. Download comprehensive report

## 🎯 Match Score Interpretation

- **90-100%**: 🟢 Excellent Fit - Apply immediately
- **80-90%**: 🟡 Good Fit - Strong candidate
- **70-80%**: 🟠 Fair Fit - Consider with development
- **<70%**: 🔴 Weak Fit - Significant gaps

## 📁 Data Requirements

Place job descriptions in: `../data/jobs/`
- Supported formats: TXT files
- One job per file

## 🛠️ Troubleshooting

**Issue**: No jobs loaded  
**Solution**: Add .txt files to data/jobs/ directory

**Issue**: File upload fails  
**Solution**: Check file format (PDF, DOCX, TXT only)

**Issue**: Model loading error  
**Solution**: Install sentence-transformers: `pip install sentence-transformers`

## 🎨 Customization

Edit `job_matcher_app.py` to:
- Change matching algorithm
- Add more job sources
- Customize UI colors
- Add additional features

## 📊 Technical Details

- **Model**: all-MiniLM-L6-v2 (Sentence-BERT)
- **Embedding Size**: 384 dimensions
- **Framework**: Streamlit
- **Matching**: Cosine similarity

## 🤝 Support

For issues or questions, refer to Day 52-53 notebooks.
