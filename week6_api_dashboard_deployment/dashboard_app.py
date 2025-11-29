
"""
AI Security & Surveillance System - Web Dashboard
Day 38: Streamlit Dashboard
"""

import streamlit as st
import requests
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, Any

# Page configuration
st.set_page_config(
    page_title="AI Security System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def get_alerts(token: str, limit: int = 50) -> Optional[Dict[str, Any]]:
    """Get alerts from database."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v2/alerts",
            headers={"Authorization": f"Bearer {token}"},
            params={"limit": limit},
            timeout=5
        )

        if response.status_code == 200:
            return response.json()
        else:
            return None

    except Exception as e:
        return None

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

# ==================================================
# LOGIN PAGE
# ==================================================

def show_login_page():
    """Display login page."""
    st.title("🔒 AI Security & Surveillance System")
    st.markdown("---")

    # Check API health
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
            st.code("cd week6_api_dashboard_deployment\nuvicorn day37_database_integration:app --reload")
            return

        # Login form
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")

        col_login, col_register = st.columns(2)

        with col_login:
            if st.button("Login", type="primary", use_container_width=True):
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
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password")
                else:
                    st.warning("Please enter username and password")

        with col_register:
            st.button("Register", use_container_width=True, disabled=True)
            st.caption("Registration coming soon!")

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
            st.caption(f"Email: {st.session_state.user_data['email']}")
            if st.session_state.user_data.get('is_admin'):
                st.badge("Admin", type="success")

        st.markdown("---")

        # Navigation
        st.subheader("Navigation")
        page = st.radio(
            "Select Page:",
            ["📊 Dashboard", "🚨 Alerts", "👥 Face Database"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Logout
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.token = None
            st.session_state.username = None
            st.session_state.user_data = None
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

    # Refresh button
    if st.button("🔄 Refresh Data"):
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
        st.metric("✅ System Status", "Online")

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Recent Alerts")
        if alerts_data and alerts_data["total_alerts"] > 0:
            # Create DataFrame
            alerts_df = pd.DataFrame(alerts_data["alerts"][:10])

            # Display table
            display_df = alerts_df[["timestamp", "alert_type", "priority", "person_name"]].copy()
            display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No alerts found")

    with col2:
        st.subheader("👥 Recent Persons")
        if faces_data and faces_data["total_persons"] > 0:
            # Create DataFrame
            persons_df = pd.DataFrame(faces_data["persons"][:10])

            # Display table
            display_df = persons_df[["name", "person_id", "face_count"]].copy()
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No persons in database")

# ==================================================
# ALERTS PAGE
# ==================================================

def show_alerts_page():
    """Display alerts page."""
    st.title("🚨 Alert Management")
    st.markdown("---")

    # Get alerts
    alerts_data = get_alerts(st.session_state.token, limit=100)

    if not alerts_data or alerts_data["total_alerts"] == 0:
        st.info("📭 No alerts found")
        return

    # Display alerts
    st.subheader(f"Total Alerts: {alerts_data['total_alerts']}")

    for alert in alerts_data["alerts"]:
        with st.expander(
            f"🚨 {alert['alert_type']} - {alert['priority'].upper()} - {alert['timestamp'][:19]}"
        ):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(f"**Person:** {alert['person_name']}")
                st.write(f"**Location:** {alert['location']}")
                st.write(f"**Description:** {alert['description']}")
                st.write(f"**Acknowledged:** {'✅ Yes' if alert['acknowledged'] else '❌ No'}")
                if alert['acknowledged']:
                    st.write(f"**Acknowledged by:** {alert['acknowledged_by']}")

            with col2:
                priority_color = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🟢"
                }
                st.markdown(f"### {priority_color.get(alert['priority'], '⚪')} {alert['priority'].upper()}")

# ==================================================
# FACES PAGE
# ==================================================

def show_faces_page():
    """Display face database page."""
    st.title("👥 Face Database")
    st.markdown("---")

    # Get faces
    faces_data = get_faces(st.session_state.token)

    if not faces_data or faces_data["total_persons"] == 0:
        st.info("📭 No persons in database")
        return

    # Display persons
    st.subheader(f"Total Persons: {faces_data['total_persons']}")

    # Create DataFrame
    persons_df = pd.DataFrame(faces_data["persons"])

    # Display table
    st.dataframe(
        persons_df[["name", "person_id", "face_count", "added_date"]],
        use_container_width=True,
        hide_index=True
    )

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
