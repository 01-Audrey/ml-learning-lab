
import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Job-Resume Matcher",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2196F3;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .match-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 1rem 0;
    }
    .skill-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 15px;
        background-color: #4CAF50;
        color: white;
        font-size: 0.9rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #2196F3;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">💼 Job-Resume Matcher</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Semantic Matching with Sentence-BERT</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    matching_mode = st.radio(
        "Matching Mode",
        ["Single Resume", "Batch Processing"],
        help="Choose single resume or process multiple at once"
    )

    top_k = st.slider(
        "Number of Matches",
        min_value=1,
        max_value=10,
        value=3,
        help="How many top job matches to show"
    )

    st.markdown("---")

    st.header("📊 About")
    st.markdown("""
    This tool uses **Sentence-BERT** for semantic similarity matching.

    **Features:**
    - PDF, DOCX, TXT support
    - Real-time matching
    - Skill gap analysis
    - Downloadable reports
    - Batch processing

    **Model:** all-MiniLM-L6-v2  
    **Accuracy:** Semantic matching
    """)

# Cache model loading
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_jobs():
    """Load job descriptions"""
    jobs = {}
    job_dir = 'data/jobs'

    if os.path.exists(job_dir):
        for filename in os.listdir(job_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(job_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    jobs[filename] = f.read()

    return jobs

# Helper functions
def parse_uploaded_file(uploaded_file):
    """Parse uploaded resume file"""
    try:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        if file_extension == '.txt':
            return uploaded_file.read().decode('utf-8')
        elif file_extension == '.pdf':
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        elif file_extension == '.docx':
            import docx
            doc = docx.Document(uploaded_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return None
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return None

def extract_keywords(text):
    """Extract skills from text"""
    keywords = [
        'python', 'pytorch', 'tensorflow', 'machine learning', 'deep learning',
        'nlp', 'computer vision', 'sql', 'docker', 'kubernetes', 'aws',
        'react', 'node.js', 'java', 'c++', 'data analysis', 'statistics'
    ]

    text_lower = text.lower()
    found = [k for k in keywords if k in text_lower]
    return found

def match_resume_to_jobs(resume_text, jobs, model, top_k=3):
    """Match resume to jobs"""
    resume_emb = model.encode(resume_text, convert_to_tensor=True)

    matches = []
    for job_name, job_text in jobs.items():
        job_emb = model.encode(job_text, convert_to_tensor=True)
        similarity = util.cos_sim(resume_emb, job_emb).item()
        score = (similarity + 1) / 2 * 100

        matches.append({
            'job': job_name,
            'score': score,
            'text': job_text
        })

    matches.sort(key=lambda x: x['score'], reverse=True)
    return matches[:top_k]

# Main app
try:
    model = load_model()
    jobs = load_jobs()

    if not jobs:
        st.warning("⚠️ No job descriptions found. Please add job files to data/jobs/")
    else:
        st.success(f"✅ Loaded {len(jobs)} job descriptions")

    # Single Resume Mode
    if matching_mode == "Single Resume":
        st.header("📄 Upload Resume")

        uploaded_file = st.file_uploader(
            "Choose a resume file",
            type=['pdf', 'docx', 'txt'],
            help="Supported formats: PDF, DOCX, TXT"
        )

        if uploaded_file is not None:
            # Parse file
            with st.spinner("📖 Parsing resume..."):
                resume_text = parse_uploaded_file(uploaded_file)

            if resume_text:
                # Display resume preview
                with st.expander("📄 Resume Preview", expanded=False):
                    st.text_area("Resume Content", resume_text[:1000] + "...", height=200)

                # Extract skills
                resume_skills = extract_keywords(resume_text)

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.metric("Skills Detected", len(resume_skills))

                    if resume_skills:
                        st.markdown("**Detected Skills:**")
                        for skill in resume_skills[:10]:
                            st.markdown(f'<span class="skill-badge">{skill}</span>', unsafe_allow_html=True)

                # Match button
                if st.button("🔍 Find Matching Jobs", type="primary"):
                    with st.spinner("🤖 AI is analyzing your resume..."):
                        matches = match_resume_to_jobs(resume_text, jobs, model, top_k)

                    st.success("✅ Matching complete!")

                    # Display matches
                    st.header(f"🎯 Top {len(matches)} Job Matches")

                    for i, match in enumerate(matches, 1):
                        score = match['score']

                        if score >= 90:
                            color = "#4CAF50"
                            quality = "🟢 EXCELLENT FIT"
                        elif score >= 80:
                            color = "#FFC107"
                            quality = "🟡 GOOD FIT"
                        elif score >= 70:
                            color = "#FF9800"
                            quality = "🟠 FAIR FIT"
                        else:
                            color = "#F44336"
                            quality = "🔴 WEAK FIT"

                        with st.container():
                            st.markdown(f"### Rank {i}: {match['job']}")

                            col1, col2, col3 = st.columns([2, 2, 3])

                            with col1:
                                st.metric("Match Score", f"{score:.1f}%")

                            with col2:
                                st.markdown(f"**Assessment:** {quality}")

                            with col3:
                                # Progress bar
                                st.progress(int(score))

                            # Job skills
                            job_skills = extract_keywords(match['text'])
                            matching_skills = set(resume_skills).intersection(set(job_skills))
                            missing_skills = set(job_skills) - set(resume_skills)

                            col1, col2 = st.columns(2)

                            with col1:
                                if matching_skills:
                                    st.markdown(f"**✅ You have:** {', '.join(list(matching_skills)[:5])}")

                            with col2:
                                if missing_skills:
                                    st.markdown(f"**⚠️ Missing:** {', '.join(list(missing_skills)[:3])}")

                            st.markdown("---")

                    # Download report
                    report_df = pd.DataFrame([
                        {
                            'Rank': i,
                            'Job': m['job'],
                            'Match Score': f"{m['score']:.2f}%"
                        }
                        for i, m in enumerate(matches, 1)
                    ])

                    csv = report_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Match Report (CSV)",
                        csv,
                        "job_matches.csv",
                        "text/csv",
                        key='download-csv'
                    )

    # Batch Processing Mode
    else:
        st.header("📦 Batch Resume Processing")
        st.info("💡 Upload multiple resumes at once for batch processing")

        uploaded_files = st.file_uploader(
            "Choose resume files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload multiple resume files"
        )

        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files uploaded")

            if st.button("🚀 Process All Resumes", type="primary"):
                results = []

                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")

                    resume_text = parse_uploaded_file(file)

                    if resume_text:
                        matches = match_resume_to_jobs(resume_text, jobs, model, top_k=1)

                        if matches:
                            results.append({
                                'Resume': file.name,
                                'Best Match': matches[0]['job'],
                                'Score': f"{matches[0]['score']:.2f}%"
                            })

                    progress_bar.progress((i + 1) / len(uploaded_files))

                status_text.text("✅ Processing complete!")

                # Display results
                st.header("📊 Batch Processing Results")

                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)

                # Download
                csv = df_results.to_csv(index=False)
                st.download_button(
                    "📥 Download Batch Report",
                    csv,
                    "batch_results.csv",
                    "text/csv"
                )

except Exception as e:
    st.error(f"❌ Error: {e}")
    st.info("Please ensure all required files are in place and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Built with ❤️ using Sentence-BERT | Day 53 of ML Learning Journey
</div>
""", unsafe_allow_html=True)
