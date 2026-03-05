
"""
AI Security & Surveillance System - Enhanced Dashboard
Day 39: Alert Management & Face Database Enhancements
"""

import streamlit as st
import requests
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, Any, List
import base64
from io import BytesIO
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="AI Security System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .alert-critical {
        border-left: 5px solid #ff4444;
        padding-left: 10px;
    }
    .alert-high {
        border-left: 5px solid #ff8800;
        padding-left: 10px;
    }
    .alert-medium {
        border-left: 5px solid #ffbb33;
        padding-left: 10px;
    }
    .alert-low {
        border-left: 5px solid #00C851;
        padding-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# ==================================================
# API HELPER FUNCTIONS
# ==================================================

def login(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Login and get JWT token."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v2/token",
            data={"username": username, "password": password},
            timeout=5
        )

        if response.status_code == 200:
            return response.json()
        else:
            return None

    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to API server. Is it running?")
        return None
    except Exception as e:
        st.error(f"❌ Login error: {e}")
        return None

def get_current_user(token: str) -> Optional[Dict[str, Any]]:
    """Get current user information."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v2/users/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5
        )

        if response.status_code == 200:
            return response.json()
        else:
            return None

    except Exception as e:
        return None

def get_faces(token: str) -> Optional[Dict[str, Any]]:
    """Get all persons from face database."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v2/faces",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5
        )

        if response.status_code == 200:
            return response.json()
        else:
            return None

    except Exception as e:
        return None

def get_alerts(token: str, limit: int = 100, priority: str = None, acknowledged: bool = None) -> Optional[Dict[str, Any]]:
    """Get alerts from database."""
    try:
        params = {"limit": limit}
        if priority:
            params["priority"] = priority
        if acknowledged is not None:
            params["acknowledged"] = acknowledged

        response = requests.get(
            f"{API_BASE_URL}/api/v2/alerts",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=5
        )

        if response.status_code == 200:
            return response.json()
        else:
            return None

    except Exception as e:
        return None

def acknowledge_alert(token: str, alert_id: int) -> bool:
    """Acknowledge an alert."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v2/alerts/acknowledge",
            headers={"Authorization": f"Bearer {token}"},
            params={"alert_id": alert_id},
            timeout=5
        )

        return response.status_code == 200

    except Exception as e:
        st.error(f"❌ Error acknowledging alert: {e}")
        return False

def add_person(token: str, person_id: str, name: str, metadata: dict = None) -> bool:
    """Add new person to database."""
    try:
        params = {"person_id": person_id, "name": name}
        if metadata:
            params["metadata"] = metadata

        response = requests.post(
            f"{API_BASE_URL}/api/v2/faces",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=5
        )

        return response.status_code == 200

    except Exception as e:
        st.error(f"❌ Error adding person: {e}")
        return False

def delete_person(token: str, person_id: str) -> bool:
    """Delete person from database."""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/api/v2/faces/{person_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5
        )

        return response.status_code == 200

    except Exception as e:
        st.error(f"❌ Error deleting person: {e}")
        return False

def check_api_health() -> bool:
    """Check if API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v2/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# ==================================================
# SESSION STATE INITIALIZATION
# ==================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.token = None
    st.session_state.username = None
    st.session_state.user_data = None
    st.session_state.auto_refresh = False
    st.session_state.refresh_interval = 30

# ==================================================
# LOGIN PAGE
# ==================================================

def show_login_page():
    """Display login page."""
    st.title("🔒 AI Security & Surveillance System")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.subheader("Login")

        # API status
        api_healthy = check_api_health()
        if api_healthy:
            st.success("✅ API Server: Online")
        else:
            st.error("❌ API Server: Offline")
            st.warning("Please start the API server first!")
            st.code("uvicorn day37_database_integration:app --reload")
            return

        # Login form
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")

            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

            if submitted:
                if username and password:
                    with st.spinner("Logging in..."):
                        token_data = login(username, password)

                        if token_data:
                            st.session_state.token = token_data["access_token"]
                            st.session_state.username = username
                            st.session_state.authenticated = True

                            # Get user data
                            user_data = get_current_user(st.session_state.token)
                            st.session_state.user_data = user_data

                            st.success("✅ Login successful!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password")
                else:
                    st.warning("Please enter username and password")

# ==================================================
# MAIN DASHBOARD
# ==================================================

def show_dashboard():
    """Display main dashboard."""

    # Sidebar
    with st.sidebar:
        st.title("🔒 Security System")
        st.markdown("---")

        # User info
        if st.session_state.user_data:
            st.subheader(f"👤 {st.session_state.user_data['username']}")
            st.caption(f"📧 {st.session_state.user_data['email']}")
            if st.session_state.user_data.get('is_admin'):
                st.success("👑 Admin")

        st.markdown("---")

        # Navigation
        st.subheader("📑 Navigation")
        page = st.radio(
            "Go to:",
            ["📊 Dashboard", "🚨 Alerts", "👥 Face Database"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Auto-refresh toggle
        st.subheader("⚙️ Settings")
        st.session_state.auto_refresh = st.checkbox("Auto-refresh", value=st.session_state.auto_refresh)
        if st.session_state.auto_refresh:
            st.session_state.refresh_interval = st.slider("Refresh interval (seconds)", 10, 60, 30)

        st.markdown("---")

        # Logout
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.token = None
            st.session_state.username = None
            st.session_state.user_data = None
            st.rerun()

    # Auto-refresh
    if st.session_state.auto_refresh:
        time.sleep(st.session_state.refresh_interval)
        st.rerun()

    # Main content
    if page == "📊 Dashboard":
        show_dashboard_page()
    elif page == "🚨 Alerts":
        show_alerts_page()
    elif page == "👥 Face Database":
        show_faces_page()

# ==================================================
# DASHBOARD PAGE
# ==================================================

def show_dashboard_page():
    """Display main dashboard page."""
    st.title("📊 System Dashboard")
    st.markdown("---")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Real-Time Overview")
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    # Get data
    faces_data = get_faces(st.session_state.token)
    alerts_data = get_alerts(st.session_state.token, limit=100)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_persons = faces_data["total_persons"] if faces_data else 0
        st.metric("👥 Total Persons", total_persons)

    with col2:
        total_alerts = alerts_data["total_alerts"] if alerts_data else 0
        st.metric("🚨 Total Alerts", total_alerts)

    with col3:
        if alerts_data:
            unack = sum(1 for a in alerts_data["alerts"] if not a["acknowledged"])
            st.metric("⚠️ Unacknowledged", unack)
        else:
            st.metric("⚠️ Unacknowledged", 0)

    with col4:
        st.metric("✅ System Status", "Online", delta="Healthy")

    st.markdown("---")

    # Recent data
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Recent Alerts")
        if alerts_data and alerts_data["total_alerts"] > 0:
            recent_alerts = alerts_data["alerts"][:10]
            for alert in recent_alerts:
                priority_class = f"alert-{alert['priority']}"
                with st.container():
                    st.markdown(f'<div class="{priority_class}">', unsafe_allow_html=True)
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.write(f"**{alert['alert_type']}** - {alert['person_name']}")
                        st.caption(f"{alert['timestamp'][:19]} | {alert['priority'].upper()}")
                    with col_b:
                        if alert['acknowledged']:
                            st.success("✅ Ack")
                        else:
                            st.warning("⚠️ New")
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No recent alerts")

    with col2:
        st.subheader("👥 Recent Persons")
        if faces_data and faces_data["total_persons"] > 0:
            persons = faces_data["persons"][:10]
            for person in persons:
                st.write(f"**{person['name']}** ({person['person_id']}) - {person['face_count']} face(s)")
        else:
            st.info("No persons in database")

# ==================================================
# ENHANCED ALERTS PAGE
# ==================================================

def show_alerts_page():
    """Display enhanced alerts page with acknowledgment."""
    st.title("🚨 Alert Management")
    st.markdown("---")

    # Filters
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

    with col1:
        priority_filter = st.selectbox(
            "Filter by Priority",
            ["All", "critical", "high", "medium", "low"]
        )

    with col2:
        ack_filter = st.selectbox(
            "Filter by Status",
            ["All", "Acknowledged", "Unacknowledged"]
        )

    with col3:
        limit = st.number_input("Max Alerts", min_value=10, max_value=200, value=50)

    with col4:
        if st.button("🔄 Refresh"):
            st.rerun()

    # Get alerts with filters
    priority_param = None if priority_filter == "All" else priority_filter
    ack_param = None if ack_filter == "All" else (ack_filter == "Acknowledged")

    alerts_data = get_alerts(st.session_state.token, limit=limit, priority=priority_param, acknowledged=ack_param)

    if not alerts_data or alerts_data["total_alerts"] == 0:
        st.info("📭 No alerts found")
        return

    # Display alerts
    st.subheader(f"Total Alerts: {alerts_data['total_alerts']}")

    for alert in alerts_data["alerts"]:
        priority_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}

        with st.expander(
            f"{priority_emoji.get(alert['priority'], '⚪')} {alert['alert_type']} - {alert['priority'].upper()} - {alert['timestamp'][:19]}",
            expanded=not alert['acknowledged']
        ):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(f"**Person:** {alert['person_name']}")
                st.write(f"**Location:** {alert['location']}")
                st.write(f"**Description:** {alert['description']}")

                if alert['acknowledged']:
                    st.success(f"✅ Acknowledged by {alert['acknowledged_by']}")
                else:
                    st.warning("⚠️ Not yet acknowledged")

            with col2:
                st.markdown(f"### {priority_emoji.get(alert['priority'], '⚪')}")
                st.write(alert['priority'].upper())

                # Acknowledge button
                if not alert['acknowledged']:
                    if st.button(f"✅ Acknowledge", key=f"ack_{alert['alert_id']}", type="primary"):
                        if acknowledge_alert(st.session_state.token, alert['alert_id']):
                            st.success("Alert acknowledged!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("Failed to acknowledge alert")

# ==================================================
# FACES PAGE
# ==================================================

def show_faces_page():
    """Display face database page."""
    st.title("👥 Face Database Management")
    st.markdown("---")

    # Add Person Section (collapsible)
    with st.expander("➕ Add New Person", expanded=False):
        with st.form("add_person_form"):
            col1, col2 = st.columns(2)

            with col1:
                person_id = st.text_input("Person ID", placeholder="e.g., john_doe")
                name = st.text_input("Name", placeholder="e.g., John Doe")

            with col2:
                department = st.text_input("Department (optional)", placeholder="e.g., Engineering")
                role = st.text_input("Role (optional)", placeholder="e.g., Engineer")

            submitted = st.form_submit_button("Add Person", type="primary")

            if submitted:
                if person_id and name:
                    metadata = {}
                    if department:
                        metadata["department"] = department
                    if role:
                        metadata["role"] = role

                    if add_person(st.session_state.token, person_id, name, metadata if metadata else None):
                        st.success(f"✅ Person '{name}' added successfully!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed to add person")
                else:
                    st.warning("Please fill in Person ID and Name")

    st.markdown("---")

    # Get faces
    faces_data = get_faces(st.session_state.token)

    if not faces_data or faces_data["total_persons"] == 0:
        st.info("📭 No persons in database")
        return

    # Display persons
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"Total Persons: {faces_data['total_persons']}")
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()

    # Search
    search = st.text_input("🔍 Search by name", placeholder="Enter name to search...")

    # Filter persons
    persons = faces_data["persons"]
    if search:
        persons = [p for p in persons if search.lower() in p["name"].lower()]

    # Display as cards
    for person in persons:
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])

            with col1:
                st.write(f"**{person['name']}**")
                st.caption(f"ID: {person['person_id']}")

            with col2:
                st.write(f"📸 {person['face_count']} face(s)")
                st.caption(f"Added: {person['added_date'][:10]}")

            with col3:
                # Delete button (admin only)
                if st.session_state.user_data.get('is_admin'):
                    if st.button("🗑️ Delete", key=f"del_{person['person_id']}", type="secondary"):
                        if delete_person(st.session_state.token, person['person_id']):
                            st.success(f"Person '{person['name']}' deleted!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("Failed to delete person")

            st.markdown("---")

# ==================================================
# MAIN APP
# ==================================================

def main():
    """Main application entry point."""
    if not st.session_state.authenticated:
        show_login_page()
    else:
        show_dashboard()

if __name__ == "__main__":
    main()
