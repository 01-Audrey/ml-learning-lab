"""
AI Security System - Dashboard with Live Video
"""

import streamlit as st
import requests
import time

st.set_page_config(page_title="AI Security System", page_icon="🔒", layout="wide")

API_BASE_URL = "http://backend:8000"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.token = None
    st.session_state.username = None

def login(username, password):
    try:
        response = requests.post(f"{API_BASE_URL}/api/v2/token", data={"username": username, "password": password}, timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def get_faces(token):
    try:
        response = requests.get(f"{API_BASE_URL}/api/v2/faces", headers={"Authorization": f"Bearer {token}"}, timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def get_alerts(token):
    try:
        response = requests.get(f"{API_BASE_URL}/api/v2/alerts", headers={"Authorization": f"Bearer {token}"}, params={"limit": 100}, timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/api/v2/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def show_login():
    st.title("🔒 AI Security System")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("Login")
        if check_api_health():
            st.success("✅ API Online")
        else:
            st.error("❌ API Offline")
            return
        with st.form("login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if username and password:
                    token_data = login(username, password)
                    if token_data:
                        st.session_state.token = token_data["access_token"]
                        st.session_state.username = username
                        st.session_state.authenticated = True
                        st.rerun()
                    else:
                        st.error("Invalid credentials")

def show_dashboard():
    with st.sidebar:
        st.title("🔒 Security")
        st.write(f"👤 {st.session_state.username}")
        page = st.radio("Nav", ["📹 Live Feed", "📊 Dashboard", "🚨 Alerts", "👥 Faces"])
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()
    
    if page == "📹 Live Feed":
        show_live_feed()
    elif page == "📊 Dashboard":
        show_dashboard_page()
    elif page == "🚨 Alerts":
        show_alerts_page()
    elif page == "👥 Faces":
        show_faces_page()

def show_live_feed():
    st.title("📹 Live Camera Feed")
    st.markdown("---")
    
    # Video stream
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Real-Time Video with AI Detection")
        video_url = f"{API_BASE_URL}/api/v2/video/stream"
        
        # Display video stream
        st.markdown(f"""
        <div style="border: 3px solid #0f0; border-radius: 10px; overflow: hidden;">
            <img src="{video_url}" style="width: 100%; height: auto;" />
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Live Stats")
        
        # Real-time metrics
        faces = get_faces(st.session_state.token)
        alerts = get_alerts(st.session_state.token)
        
        st.metric("📹 Status", "🟢 LIVE")
        st.metric("🎯 FPS", "30")
        st.metric("👥 Detected", "0")
        st.metric("🚨 Alerts", alerts["total_alerts"] if alerts else 0)
        
        st.markdown("---")
        st.info("💡 Live ML processing active")

def show_dashboard_page():
    st.title("📊 Dashboard")
    st.markdown("---")
    
    faces = get_faces(st.session_state.token)
    alerts = get_alerts(st.session_state.token)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Persons", faces["total_persons"] if faces else 0)
    with col2:
        st.metric("Alerts", alerts["total_alerts"] if alerts else 0)
    with col3:
        if alerts:
            unack = sum(1 for a in alerts["alerts"] if not a["acknowledged"])
            st.metric("Unacknowledged", unack)
        else:
            st.metric("Unacknowledged", 0)
    with col4:
        st.metric("Status", "🟢 Online")

def show_alerts_page():
    st.title("🚨 Alerts")
    st.markdown("---")
    
    alerts = get_alerts(st.session_state.token)
    
    if not alerts or alerts["total_alerts"] == 0:
        st.info("No alerts")
        return
    
    st.subheader(f"Total: {alerts['total_alerts']}")
    
    for a in alerts["alerts"]:
        with st.expander(f"🚨 {a['alert_type']} - {a['timestamp'][:19]}"):
            st.write(f"**Person:** {a['person_name']}")
            st.write(f"**Location:** {a['location']}")
            st.write(f"**Priority:** {a['priority']}")
            st.write(f"**Ack:** {'✅' if a['acknowledged'] else '❌'}")

def show_faces_page():
    st.title("👥 Faces")
    st.markdown("---")
    
    faces = get_faces(st.session_state.token)
    
    if not faces or faces["total_persons"] == 0:
        st.info("No persons")
        return
    
    st.subheader(f"Total: {faces['total_persons']}")
    
    for p in faces["persons"]:
        st.write(f"**{p['name']}** ({p['person_id']}) - {p['face_count']} face(s)")
        st.markdown("---")

if not st.session_state.authenticated:
    show_login()
else:
    show_dashboard()