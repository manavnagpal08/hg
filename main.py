# ============================================
# ScreenerPro – AI Resume Screening App (Redesigned UI)
# ============================================
import streamlit as st
import json, os, datetime
from login import login_section, is_current_user_admin

# --- Config ---
st.set_page_config("ScreenerPro Dashboard", layout="wide", page_icon="🧠")

# --- Global Styling ---
st.markdown("""
<style>
    /* Hide default menu/footer */
    #MainMenu, header, footer {visibility: hidden;}

    /* Font and layout */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #F8FAFC;
        color: #222;
    }

    /* Top Header Bar */
    .top-header {
        background: #ffffff;
        padding: 1rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border-radius: 0 0 12px 12px;
        margin-bottom: 1rem;
    }
    .top-header h1 {
        margin: 0;
        font-size: 1.8rem;
        color: #00cec9;
    }
    .top-header .profile {
        font-weight: 500;
        color: #444;
    }

    /* Metrics */
    .metric-card {
        background: #fff;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: 0.3s ease;
    }
    .metric-card:hover {
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        transform: translateY(-4px);
    }
    .metric-card h2 {
        font-size: 1.3rem;
        color: #00cec9;
    }
    .metric-card p {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.4rem;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.image("logo.png", width=180)
st.sidebar.markdown("### 🧭 Navigation")
nav = st.sidebar.radio("Go to", [
    "🏠 Dashboard",
    "🧠 Resume Screener",
    "📁 Job Descriptions",
    "📊 Analytics",
    "📤 Email Candidates",
    "⚙️ Admin Tools",
    "🚪 Logout"
])

# --- Auth ---
if not login_section():
    st.stop()
username = st.session_state.get("username", "User")

# --- Header Bar ---
st.markdown(f"""
    <div class="top-header">
        <h1>ScreenerPro</h1>
        <div class="profile">
            👋 Hello, <b>{username}</b> | <a href="?nav=Logout">Logout</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- Dashboard Page ---
if nav == "🏠 Dashboard":
    st.markdown("## 📊 Overview Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>Resumes</h2>
            <p>125</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>Shortlisted</h2>
            <p>38</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>Avg. Score</h2>
            <p>82%</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h2>JDs Uploaded</h2>
            <p>6</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("Select 'Resume Screener' from sidebar to get started.")

# --- Other Pages Placeholder ---
elif nav == "🧠 Resume Screener":
    st.markdown("## Resume Screener")
    st.info("Your screener logic will load here.")

elif nav == "📁 Job Descriptions":
    st.markdown("## Manage Job Descriptions")
    st.info("Upload or edit job roles here.")

elif nav == "📊 Analytics":
    st.markdown("## Screening Analytics")
    st.info("Graphs and insights will be shown here.")

elif nav == "📤 Email Candidates":
    st.markdown("## Email Shortlisted Candidates")
    st.info("Send emails manually or in bulk.")

elif nav == "⚙️ Admin Tools":
    if is_current_user_admin():
        st.markdown("## Admin Tools")
        st.success("You are logged in as admin.")
    else:
        st.error("Access Denied. Only admins can view this page.")

elif nav == "🚪 Logout":
    st.session_state.authenticated = False
    st.success("You’ve been logged out.")
    st.rerun()
