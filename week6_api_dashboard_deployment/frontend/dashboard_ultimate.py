
"""
AI Security & Surveillance System - Ultimate Dashboard
Day 40: Advanced Analytics & Visualizations
"""

import streamlit as st
import requests
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, Any, List
import json
import time
import numpy as np

# Page configuration
st.set_page_config(
    page_title="AI Security System - Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
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
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# ==================================================
# API HELPER FUNCTIONS (Same as Day 39)
# ==================================================

def login(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Login and get JWT token."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v2/token",
            data={"username": username, "password": password},
            timeout=5
        )
        return response.json() if response.status_code == 200 else None
    except:
        return None

def get_current_user(token: str) -> Optional[Dict[str, Any]]:
    """Get current user information."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v2/users/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5
        )
        return response.json() if response.status_code == 200 else None
    except:
        return None

def get_faces(token: str) -> Optional[Dict[str, Any]]:
    """Get all persons from face database."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v2/faces",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5
        )
        return response.json() if response.status_code == 200 else None
    except:
        return None

def get_alerts(token: str, limit: int = 500, priority: str = None, acknowledged: bool = None) -> Optional[Dict[str, Any]]:
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
        return response.json() if response.status_code == 200 else None
    except:
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
    except:
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
    except:
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
    except:
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
    st.session_state.date_range = 7  # Days

# ==================================================
# ANALYTICS HELPER FUNCTIONS
# ==================================================

def create_temporal_heatmap(alerts_df):
    """Create heat map showing alerts by hour and day of week."""
    if len(alerts_df) == 0:
        return None

    # Extract hour and day of week
    alerts_df["hour"] = alerts_df["timestamp"].dt.hour
    alerts_df["day_of_week"] = alerts_df["timestamp"].dt.day_name()

    # Create pivot table
    heatmap_data = alerts_df.pivot_table(
        index="day_of_week",
        columns="hour",
        values="alert_id",
        aggfunc="count",
        fill_value=0
    )

    # Reorder days
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_data = heatmap_data.reindex([d for d in days_order if d in heatmap_data.index], fill_value=0)

    # Create heat map
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=list(range(24)),
        y=heatmap_data.index,
        colorscale="Reds",
        colorbar=dict(title="Alert Count"),
        hoverongaps=False,
        hovertemplate="Day: %{y}<br>Hour: %{x}:00<br>Alerts: %{z}<extra></extra>"
    ))

    fig.update_layout(
        title="🔥 Alert Heat Map: Day of Week × Hour of Day",
        xaxis_title="Hour of Day (0-23)",
        yaxis_title="Day of Week",
        height=400
    )

    return fig

def create_hourly_distribution(alerts_df):
    """Create hourly distribution bar chart."""
    if len(alerts_df) == 0:
        return None

    alerts_df["hour"] = alerts_df["timestamp"].dt.hour
    hourly_counts = alerts_df.groupby("hour").size()

    fig = go.Figure(data=go.Bar(
        x=list(range(24)),
        y=[hourly_counts.get(h, 0) for h in range(24)],
        marker_color="#ff4444",
        hovertemplate="Hour: %{x}:00<br>Alerts: %{y}<extra></extra>"
    ))

    fig.update_layout(
        title="📊 Hourly Alert Distribution",
        xaxis_title="Hour of Day",
        yaxis_title="Alert Count",
        height=300
    )

    return fig

def create_daily_trend(alerts_df):
    """Create daily trend line chart."""
    if len(alerts_df) == 0:
        return None

    alerts_df["date"] = alerts_df["timestamp"].dt.date
    daily_counts = alerts_df.groupby("date").size().reset_index(name="count")

    fig = go.Figure()

    # Line
    fig.add_trace(go.Scatter(
        x=daily_counts["date"],
        y=daily_counts["count"],
        mode='lines+markers',
        name='Alerts',
        line=dict(color='#ff4444', width=3),
        marker=dict(size=8)
    ))

    # Trend line
    if len(daily_counts) > 1:
        x_numeric = np.arange(len(daily_counts))
        z = np.polyfit(x_numeric, daily_counts["count"], 1)
        p = np.poly1d(z)

        fig.add_trace(go.Scatter(
            x=daily_counts["date"],
            y=p(x_numeric),
            mode='lines',
            name='Trend',
            line=dict(color='#00C851', width=2, dash='dash')
        ))

    fig.update_layout(
        title="📈 Daily Alert Trend",
        xaxis_title="Date",
        yaxis_title="Alert Count",
        height=400,
        hovermode='x unified'
    ))

    return fig

def create_priority_timeline(alerts_df):
    """Create stacked area chart of priorities over time."""
    if len(alerts_df) == 0:
        return None

    alerts_df["date"] = alerts_df["timestamp"].dt.date

    priority_daily = alerts_df.pivot_table(
        index="date",
        columns="priority",
        values="alert_id",
        aggfunc="count",
        fill_value=0
    )

    fig = go.Figure()

    colors = {
        "critical": "#ff4444",
        "high": "#ff8800",
        "medium": "#ffbb33",
        "low": "#00C851"
    }

    for priority in ["low", "medium", "high", "critical"]:
        if priority in priority_daily.columns:
            fig.add_trace(go.Scatter(
                x=priority_daily.index,
                y=priority_daily[priority],
                mode='lines',
                name=priority.capitalize(),
                stackgroup='one',
                line=dict(width=0.5, color=colors.get(priority, "#999")),
                fillcolor=colors.get(priority, "#999")
            ))

    fig.update_layout(
        title="📊 Priority Distribution Over Time",
        xaxis_title="Date",
        yaxis_title="Alert Count",
        height=400,
        hovermode='x unified'
    )

    return fig

# ==================================================
# LOGIN PAGE (Same as Day 39)
# ==================================================

def show_login_page():
    """Display login page."""
    st.title("🔒 AI Security & Surveillance System")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.subheader("Login")

        api_healthy = check_api_health()
        if api_healthy:
            st.success("✅ API Server: Online")
        else:
            st.error("❌ API Server: Offline")
            st.warning("Please start the API server first!")
            st.code("uvicorn day37_database_integration:app --reload")
            return

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")

            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

            if submitted and username and password:
                with st.spinner("Logging in..."):
                    token_data = login(username, password)

                    if token_data:
                        st.session_state.token = token_data["access_token"]
                        st.session_state.username = username
                        st.session_state.authenticated = True
                        st.session_state.user_data = get_current_user(st.session_state.token)

                        st.success("✅ Login successful!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")

# ==================================================
# MAIN DASHBOARD (Continued in next cell...)
# ==================================================

# ==================================================
# MAIN DASHBOARD NAVIGATION
# ==================================================

def show_dashboard():
    """Display main dashboard with navigation."""

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
            ["📊 Overview", "🚨 Alerts", "👥 Faces", "📈 Analytics", "⚡ Performance"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Settings
        st.subheader("⚙️ Settings")
        st.session_state.auto_refresh = st.checkbox("Auto-refresh", value=st.session_state.auto_refresh)
        if st.session_state.auto_refresh:
            st.session_state.refresh_interval = st.slider("Refresh (sec)", 10, 60, 30)

        st.session_state.date_range = st.slider("Date Range (days)", 1, 30, 7)

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

    # Route to pages
    if page == "📊 Overview":
        show_overview_page()
    elif page == "🚨 Alerts":
        show_alerts_page()
    elif page == "👥 Faces":
        show_faces_page()
    elif page == "📈 Analytics":
        show_analytics_page()
    elif page == "⚡ Performance":
        show_performance_page()

# ==================================================
# OVERVIEW PAGE
# ==================================================

def show_overview_page():
    """Display overview dashboard."""
    st.title("📊 System Overview")
    st.markdown("---")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"📅 Last {st.session_state.date_range} Days")
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    # Get data
    faces_data = get_faces(st.session_state.token)
    alerts_data = get_alerts(st.session_state.token, limit=500)

    if not alerts_data or alerts_data["total_alerts"] == 0:
        st.info("📭 No data available")
        return

    # Convert to DataFrame
    alerts_df = pd.DataFrame(alerts_data["alerts"])
    alerts_df["timestamp"] = pd.to_datetime(alerts_df["timestamp"])

    # Filter by date range
    cutoff_date = datetime.now() - timedelta(days=st.session_state.date_range)
    alerts_df = alerts_df[alerts_df["timestamp"] >= cutoff_date]

    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("🚨 Total Alerts", len(alerts_df))

    with col2:
        critical = len(alerts_df[alerts_df["priority"] == "critical"])
        st.metric("🔴 Critical", critical)

    with col3:
        high = len(alerts_df[alerts_df["priority"] == "high"])
        st.metric("🟠 High", high)

    with col4:
        ack_rate = (alerts_df["acknowledged"].sum() / len(alerts_df)) * 100 if len(alerts_df) > 0 else 0
        st.metric("✅ Ack Rate", f"{ack_rate:.1f}%")

    with col5:
        avg_per_day = len(alerts_df) / st.session_state.date_range
        st.metric("📈 Avg/Day", f"{avg_per_day:.1f}")

    st.markdown("---")

    # Charts Row 1
    col1, col2 = st.columns(2)

    with col1:
        fig = create_daily_trend(alerts_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = create_priority_timeline(alerts_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Quick Stats
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🔝 Top Alert Types")
        type_counts = alerts_df["alert_type"].value_counts().head(5)
        for alert_type, count in type_counts.items():
            st.write(f"**{alert_type}**: {count}")

    with col2:
        st.subheader("⏰ Peak Hours")
        alerts_df["hour"] = alerts_df["timestamp"].dt.hour
        hourly = alerts_df.groupby("hour").size().sort_values(ascending=False).head(5)
        for hour, count in hourly.items():
            st.write(f"**{hour}:00-{hour}:59**: {count} alerts")

    with col3:
        st.subheader("📅 Busiest Days")
        alerts_df["date"] = alerts_df["timestamp"].dt.date
        daily = alerts_df.groupby("date").size().sort_values(ascending=False).head(5)
        for date, count in daily.items():
            st.write(f"**{date}**: {count} alerts")

# ==================================================
# ALERTS PAGE (Same as Day 39 but compact)
# ==================================================

def show_alerts_page():
    """Display alerts page."""
    st.title("🚨 Alert Management")
    st.markdown("---")

    # Filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        priority_filter = st.selectbox("Priority", ["All", "critical", "high", "medium", "low"])
    with col2:
        ack_filter = st.selectbox("Status", ["All", "Acknowledged", "Unacknowledged"])
    with col3:
        limit = st.number_input("Limit", 10, 200, 50)
    with col4:
        if st.button("🔄 Refresh"):
            st.rerun()

    # Get alerts
    priority_param = None if priority_filter == "All" else priority_filter
    ack_param = None if ack_filter == "All" else (ack_filter == "Acknowledged")
    alerts_data = get_alerts(st.session_state.token, limit=limit, priority=priority_param, acknowledged=ack_param)

    if not alerts_data or alerts_data["total_alerts"] == 0:
        st.info("📭 No alerts found")
        return

    st.subheader(f"Total: {alerts_data['total_alerts']}")

    priority_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}

    for alert in alerts_data["alerts"]:
        with st.expander(
            f"{priority_emoji.get(alert['priority'], '⚪')} {alert['alert_type']} - {alert['timestamp'][:19]}",
            expanded=False
        ):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(f"**Person:** {alert['person_name']}")
                st.write(f"**Location:** {alert['location']}")
                st.write(f"**Priority:** {alert['priority'].upper()}")
                if alert['acknowledged']:
                    st.success(f"✅ Ack by {alert['acknowledged_by']}")

            with col2:
                if not alert['acknowledged']:
                    if st.button("✅ Ack", key=f"ack_{alert['alert_id']}", type="primary"):
                        if acknowledge_alert(st.session_state.token, alert['alert_id']):
                            st.success("Done!")
                            time.sleep(0.5)
                            st.rerun()

# ==================================================
# FACES PAGE (Same as Day 39 but compact)
# ==================================================

def show_faces_page():
    """Display faces page."""
    st.title("👥 Face Database")
    st.markdown("---")

    # Add person
    with st.expander("➕ Add Person"):
        with st.form("add_form"):
            col1, col2 = st.columns(2)
            with col1:
                person_id = st.text_input("ID")
                name = st.text_input("Name")
            with col2:
                dept = st.text_input("Department")
                role = st.text_input("Role")

            if st.form_submit_button("Add", type="primary"):
                if person_id and name:
                    metadata = {}
                    if dept:
                        metadata["department"] = dept
                    if role:
                        metadata["role"] = role

                    if add_person(st.session_state.token, person_id, name, metadata or None):
                        st.success(f"✅ Added {name}!")
                        time.sleep(0.5)
                        st.rerun()

    st.markdown("---")

    # Get faces
    faces_data = get_faces(st.session_state.token)

    if not faces_data or faces_data["total_persons"] == 0:
        st.info("📭 No persons")
        return

    st.subheader(f"Total: {faces_data['total_persons']}")

    # Search
    search = st.text_input("🔍 Search")

    persons = faces_data["persons"]
    if search:
        persons = [p for p in persons if search.lower() in p["name"].lower()]

    # Display
    for person in persons:
        col1, col2, col3 = st.columns([3, 2, 1])

        with col1:
            st.write(f"**{person['name']}** ({person['person_id']})")
        with col2:
            st.write(f"📸 {person['face_count']} faces")
        with col3:
            if st.session_state.user_data.get('is_admin'):
                if st.button("🗑️", key=f"del_{person['person_id']}"):
                    if delete_person(st.session_state.token, person['person_id']):
                        st.success("Deleted!")
                        time.sleep(0.5)
                        st.rerun()

        st.markdown("---")

# ==================================================
# ADVANCED ANALYTICS PAGE
# ==================================================

def show_analytics_page():
    """Display advanced analytics."""
    st.title("📈 Advanced Analytics")
    st.markdown("---")

    # Get data
    alerts_data = get_alerts(st.session_state.token, limit=500)

    if not alerts_data or alerts_data["total_alerts"] == 0:
        st.info("📭 No data")
        return

    alerts_df = pd.DataFrame(alerts_data["alerts"])
    alerts_df["timestamp"] = pd.to_datetime(alerts_df["timestamp"])

    # Filter by date
    cutoff = datetime.now() - timedelta(days=st.session_state.date_range)
    alerts_df = alerts_df[alerts_df["timestamp"] >= cutoff]

    if len(alerts_df) == 0:
        st.info(f"📭 No data in last {st.session_state.date_range} days")
        return

    # Heat Map
    st.subheader("🔥 Temporal Heat Map")
    fig = create_temporal_heatmap(alerts_df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Hourly Distribution
    col1, col2 = st.columns(2)

    with col1:
        fig = create_hourly_distribution(alerts_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Priority pie
        st.subheader("🥧 Priority Breakdown")
        priority_counts = alerts_df["priority"].value_counts()
        fig = px.pie(
            values=priority_counts.values,
            names=priority_counts.index,
            color=priority_counts.index,
            color_discrete_map={
                "critical": "#ff4444",
                "high": "#ff8800",
                "medium": "#ffbb33",
                "low": "#00C851"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Statistical Summary
    st.subheader("📊 Statistical Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Alerts", len(alerts_df))
        st.metric("Unique Days", alerts_df["timestamp"].dt.date.nunique())

    with col2:
        st.metric("Mean/Day", f"{len(alerts_df) / st.session_state.date_range:.2f}")
        st.metric("Std Dev", f"{alerts_df.groupby(alerts_df['timestamp'].dt.date).size().std():.2f}")

    with col3:
        st.metric("Peak Hour", f"{alerts_df['timestamp'].dt.hour.mode()[0] if len(alerts_df) > 0 else 0}:00")
        st.metric("Min/Hour", f"{alerts_df.groupby(alerts_df['timestamp'].dt.hour).size().min()}")

    with col4:
        st.metric("Max/Hour", f"{alerts_df.groupby(alerts_df['timestamp'].dt.hour).size().max()}")
        st.metric("Ack Rate", f"{(alerts_df['acknowledged'].sum() / len(alerts_df) * 100):.1f}%")

    st.markdown("---")

    # Export
    st.subheader("📥 Export")
    col1, col2 = st.columns(2)

    with col1:
        csv = alerts_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📄 Download CSV",
            csv,
            f"analytics_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            use_container_width=True
        )

    with col2:
        stats = {
            "total": len(alerts_df),
            "date_range": st.session_state.date_range,
            "mean_per_day": len(alerts_df) / st.session_state.date_range,
            "peak_hour": int(alerts_df['timestamp'].dt.hour.mode()[0]) if len(alerts_df) > 0 else 0
        }
        json_data = json.dumps(stats, indent=2).encode('utf-8')
        st.download_button(
            "📊 Download JSON",
            json_data,
            f"stats_{datetime.now().strftime('%Y%m%d')}.json",
            "application/json",
            use_container_width=True
        )

# ==================================================
# PERFORMANCE PAGE
# ==================================================

def show_performance_page():
    """Display system performance metrics."""
    st.title("⚡ System Performance")
    st.markdown("---")

    # System Health
    st.subheader("🏥 System Health")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        api_status = check_api_health()
        st.metric("API Status", "🟢 Online" if api_status else "🔴 Offline")

    with col2:
        st.metric("Database", "🟢 Connected")

    with col3:
        st.metric("Auth", "🟢 Active")

    with col4:
        st.metric("Uptime", "99.9%")

    st.markdown("---")

    # Performance Metrics
    st.subheader("📊 Performance Metrics")

    # Simulated metrics (in production, these would be real)
    col1, col2 = st.columns(2)

    with col1:
        # Response time chart
        times = np.random.normal(50, 10, 100).tolist()
        fig = go.Figure(data=go.Scatter(
            y=times,
            mode='lines',
            line=dict(color='#00C851', width=2),
            fill='tozeroy'
        ))
        fig.update_layout(
            title="API Response Time (ms)",
            yaxis_title="Milliseconds",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Success rate
        fig = go.Figure(data=go.Indicator(
            mode="gauge+number+delta",
            value=99.5,
            delta={'reference': 99},
            title={'text': "Success Rate (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#00C851"},
                'steps': [
                    {'range': [0, 90], 'color': "#ff4444"},
                    {'range': [90, 95], 'color': "#ffbb33"},
                    {'range': [95, 100], 'color': "#E0E0E0"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 98
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Resource Usage
    st.subheader("💻 Resource Usage")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("CPU Usage", "45%", delta="-5%")

    with col2:
        st.metric("Memory", "1.2 GB", delta="+0.1 GB")

    with col3:
        st.metric("Disk I/O", "15 MB/s", delta="+2 MB/s")

# ==================================================
# MAIN APP
# ==================================================

def main():
    """Main application."""
    if not st.session_state.authenticated:
        show_login_page()
    else:
        show_dashboard()

if __name__ == "__main__":
    main()
