import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import json
import datetime # Import datetime for timestamps
import plotly.express as px # Re-added plotly.express for interactive graphs

# Import the page functions from their respective files
from login import (
    login_section, load_users, admin_registration_section,
    admin_password_reset_section, admin_disable_enable_user_section,
    is_current_user_admin
)

# Import the feedback page function
from feedback import feedback_and_help_page # NEW: Import feedback page

# --- Helper for Activity Logging ---
def log_activity(message):
    """Logs an activity with a timestamp to the session state."""
    if 'activity_log' not in st.session_state:
        st.session_state.activity_log = []
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.activity_log.insert(0, f"[{timestamp}] {message}") # Add to the beginning for most recent first
    # Keep log size manageable, e.g., last 50 activities
    st.session_state.activity_log = st.session_state.activity_log[:50]

# --- Page Config ---
st.set_page_config(page_title="ScreenerPro – AI Hiring Dashboard", layout="wide", page_icon="🧠")

# --- Dark Mode Toggle ---
dark_mode = st.sidebar.toggle("🌙 Dark Mode", key="dark_mode_main")

# --- Global Fonts & UI Styling ---
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
/* Hide GitHub fork button, Streamlit menu and footer */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}} /* Optional: hides the top bar */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {'#1E1E1E' if dark_mode else '#F0F2F6'}; /* Darker background for dark mode */
    color: {'#E0E0E0' if dark_mode else '#333333'}; /* Lighter text for dark mode */
}}
.main .block-container {{
    padding: 2rem;
    border-radius: 20px;
    background: {'#2D2D2D' if dark_mode else 'rgba(255, 255, 255, 0.96)'};
    box-shadow: 0 12px 30px rgba(0,0,0,{'0.3' if dark_mode else '0.1'});
    animation: fadeIn 0.8s ease-in-out;
}}
@keyframes fadeIn {{
    0% {{ opacity: 0; transform: translateY(20px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}
h1, h2, h3, h4, h5, h6 {{
    color: {'#00cec9' if dark_mode else '#00cec9'}; /* Consistent teal for headers */
    font-weight: 700;
}}
.dashboard-header {{
    font-size: 2.2rem;
    font-weight: 700;
    color: {'#E0E0E0' if dark_mode else '#222'};
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #00cec9;
    display: inline-block;
    margin-bottom: 2rem;
    animation: slideInLeft 0.8s ease-out;
}}
@keyframes slideInLeft {{
    0% {{ transform: translateX(-40px); opacity: 0; }}
    100% {{ transform: translateX(0); opacity: 1; }}
}}
.stMetric {{
    background-color: {'#3A3A3A' if dark_mode else '#f0f2f6'};
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,{'0.2' if dark_mode else '0.05'});
    transition: transform 0.2s ease;
}}
.stMetric:hover {{
    transform: translateY(-3px);
}}
.stMetric > div[data-testid="stMetricValue"] {{
    font-size: 2.5rem;
    font-weight: 700;
    color: {'#00cec9' if dark_mode else '#00cec9'};
}}
.stMetric > div[data-testid="stMetricLabel"] {{
    font-size: 1rem;
    color: {'#BBBBBB' if dark_mode else '#555555'};
}}
.stButton>button {{
    background-color: #00cec9;
    color: white;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    border: none;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}}
.stButton>button:hover {{
    background-color: #00b0a8;
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.15);
}}
.stExpander {{
    background-color: {'#3A3A3A' if dark_mode else '#f0f2f6'};
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,{'0.2' if dark_mode else '0.05'});
}}
.stExpander > div > div > div > p {{
    color: {'#E0E0E0' if dark_mode else '#333333'};
}}
.stExpander > div[data-testid="stExpanderToggle"] {{
    color: {'#00cec9' if dark_mode else '#00cec9'};
}}
.stExpander > div[data-testid="stExpanderToggle"] svg {{
    fill: {'#00cec9' if dark_mode else '#00cec9'};
}}
.stSelectbox > div > div {{
    background-color: {'#3A3A3A' if dark_mode else '#f0f2f6'};
    color: {'#E0E0E0' if dark_mode else '#333333'};
    border-radius: 8px;
}}
.stSelectbox > label {{
    color: {'#E0E0E0' if dark_mode else '#333333'};
}}
.stTextInput > div > div > input {{
    background-color: {'#3A3A3A' if dark_mode else '#f0f2f6'};
    color: {'#E0E0E0' if dark_mode else '#333333'};
    border-radius: 8px;
}}
.stTextInput > label {{
    color: {'#E0E0E0' if dark_mode else '#333333'};
}}
.stTextArea > div > div {{
    background-color: {'#3A3A3A' if dark_mode else '#f0f2f6'};
    color: {'#E0E0E0' if dark_mode else '#333333'};
    border-radius: 8px;
}}
.stTextArea > label {{
    color: {'#E0E0E0' if dark_mode else '#333333'};
}}
.stRadio > label {{
    color: {'#E0E0E0' if dark_mode else '#333333'};
}}
.stRadio div[role="radiogroup"] label {{
    background-color: {'#3A3A3A' if dark_mode else '#f0f2f6'};
    border-radius: 8px;
    padding: 0.5rem 1rem;
    margin: 0.2rem;
    color: {'#E0E0E0' if dark_mode else '#333333'};
}}
.stRadio div[role="radiogroup"] label:hover {{
    background-color: {'#4A4A4A' if dark_mode else '#e0e2e6'};
}}
.stRadio div[role="radiogroup"] label[data-baseweb="radio"] span:first-child {{
    background-color: {'#00cec9' if dark_mode else '#00cec9'} !important;
}}
.stCheckbox span {{
    color: {'#E0E0E0' if dark_mode else '#333333'};
}}
.stCheckbox div[data-testid="stCheckbox"] svg {{
    fill: {'#00cec9' if dark_mode else '#00cec9'};
}}
</style>
""", unsafe_allow_html=True)

# Set Matplotlib style for dark mode if active
if dark_mode:
    plt.style.use('dark_background')
    sns.set_palette("viridis") # A good palette for dark backgrounds
else:
    plt.style.use('default')
    sns.set_palette("coolwarm") # A good palette for light backgrounds


# --- Branding ---
st.sidebar.image("logo.png", width=200) # Placeholder logo
st.sidebar.title("🧠 ScreenerPro")

# --- Auth ---
if not login_section():
    st.stop()
else:
    # Log successful login
    if st.session_state.get('last_login_logged_for_user') != st.session_state.username:
        log_activity(f"User '{st.session_state.username}' logged in.")
        st.session_state.last_login_logged_for_user = st.session_state.username

# Determine if the logged-in user is an admin
is_admin = is_current_user_admin()

# Initialize comprehensive_df globally if it doesn't exist
# This ensures it's always a DataFrame, even if empty, preventing potential KeyErrors
if 'comprehensive_df' not in st.session_state:
    st.session_state['comprehensive_df'] = pd.DataFrame()

# --- Navigation Control ---
navigation_options = [
    "🏠 Dashboard", "🧠 Resume Screener", "📁 Manage JDs", "📊 Screening Analytics",
    "📤 Email Candidates", "🔍 Search Resumes", "📝 Candidate Notes", "❓ Feedback & Help"
]

if is_admin: # Only add Admin Tools if the user is an admin
    navigation_options.append("⚙️ Admin Tools")

navigation_options.append("🚪 Logout") # Always add Logout last

default_tab = st.session_state.get("tab_override", "🏠 Dashboard")

if default_tab not in navigation_options: # Handle cases where default_tab might be Admin Tools for non-admins
    default_tab = "🏠 Dashboard"

tab = st.sidebar.radio("📍 Navigate", navigation_options, index=navigation_options.index(default_tab))

if "tab_override" in st.session_state:
    del st.session_state.tab_override

# ======================
# 🏠 Dashboard Section
# ======================
if tab == "🏠 Dashboard":
    st.markdown('<div class="dashboard-header">📊 Overview Dashboard</div>', unsafe_allow_html=True)

    # Initialize metrics
    resume_count = 0
    # Create the 'data' directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
    jd_count = len([f for f in os.listdir("data") if f.endswith(".txt")])
    shortlisted = 0
    avg_score = 0.0
    df_results = pd.DataFrame()

    # Initialize cutoff_score and min_exp_required with default values to prevent NameError
    cutoff_score = st.session_state.get('screening_cutoff_score', 75)
    min_exp_required = st.session_state.get('screening_min_experience', 2)
    max_exp_allowed = st.session_state.get('screening_max_experience', 10) # Added max experience
    min_cgpa_required = st.session_state.get('screening_min_cgpa', 2.5) # Added min CGPA

    # Load results from session state
    if 'comprehensive_df' in st.session_state and not st.session_state['comprehensive_df'].empty:
        try:
            df_results = st.session_state['comprehensive_df'].copy() # Use .copy() to avoid modifying original
            resume_count = df_results["File Name"].nunique()
            
            shortlisted_df = df_results[
                (df_results["Score (%)"] >= cutoff_score) &
                (df_results["Years Experience"] >= min_exp_required) &
                (df_results["Years Experience"] <= max_exp_allowed) & # Apply max experience filter
                ((df_results['CGPA (4.0 Scale)'].isnull()) | (df_results['CGPA (4.0 Scale)'] >= min_cgpa_required)) # Apply CGPA filter
            ].copy()
            shortlisted = shortlisted_df.shape[0]
            avg_score = df_results["Score (%)"].mean()
        except Exception as e:
            st.error(f"Error processing screening results from session state: {e}")
            df_results = pd.DataFrame()
            shortlisted_df = pd.DataFrame()
    else:
        st.info("No screening results available in this session yet. Please run the Resume Screener.")
        shortlisted_df = pd.DataFrame()

    # --- Key Metrics Display ---
    st.subheader("Key Performance Indicators")
    metric_cols = st.columns(4)

    metric_cols[0].metric("Resumes Screened", resume_count, help="Total unique resumes processed in this session.")
    metric_cols[1].metric("Job Descriptions", jd_count, help="Number of job descriptions available.")
    metric_cols[2].metric("Shortlisted Candidates", shortlisted, help=f"Candidates meeting Score ≥ {cutoff_score}%, Exp {min_exp_required}-{max_exp_allowed} yrs, CGPA ≥ {min_cgpa_required} or N/A.")
    metric_cols[3].metric("Average Score", f"{avg_score:.1f}%", help="Average matching score of all screened resumes.")

    st.markdown("---")

    # --- Quick Actions ---
    st.subheader("Quick Actions")
    action_cols = st.columns(3)
    with action_cols[0]:
        if st.button("🚀 Start New Screening", key="dashboard_screener_button_large"):
            st.session_state.tab_override = '🧠 Resume Screener'
            st.rerun()
    with action_cols[1]:
        if st.button("📈 View Full Analytics", key="dashboard_analytics_button_large"):
            st.session_state.tab_override = '📊 Screening Analytics'
            st.rerun()
    with action_cols[2]:
        if st.button("📧 Email Shortlisted", key="dashboard_email_button_large"):
            st.session_state.tab_override = '📤 Email Candidates'
            st.rerun()
    
    st.markdown("---")

    # --- Dashboard Widget Customization ---
    st.subheader("⚙️ Customize Your Dashboard")
    with st.expander("Select Widgets to Display"):
        if 'dashboard_widgets' not in st.session_state:
            st.session_state.dashboard_widgets = {
                'Candidate Distribution': True,
                'Experience Distribution': True,
                'Top 5 Most Common Skills': True,
                'My Recent Screenings': True,
                'Top Performing JDs': True,
                'Pending Approvals': True,
            }

        st.session_state.dashboard_widgets['Candidate Distribution'] = st.checkbox("Candidate Quality Distribution", value=st.session_state.dashboard_widgets['Candidate Distribution'], key="widget_cand_dist")
        st.session_state.dashboard_widgets['Experience Distribution'] = st.checkbox("Experience Level Breakdown", value=st.session_state.dashboard_widgets['Experience Distribution'], key="widget_exp_dist")
        st.session_state.dashboard_widgets['Top 5 Most Common Skills'] = st.checkbox("Top 5 Matched Skills", value=st.session_state.dashboard_widgets['Top 5 Most Common Skills'], key="widget_top_skills")
        st.session_state.dashboard_widgets['My Recent Screenings'] = st.checkbox("My Recent Screenings Table", value=st.session_state.dashboard_widgets['My Recent Screenings'], key="widget_recent_screenings")
        st.session_state.dashboard_widgets['Top Performing JDs'] = st.checkbox("Top Performing Job Descriptions", value=st.session_state.dashboard_widgets['Top Performing JDs'], key="widget_top_jds")
        st.session_state.dashboard_widgets['Pending Approvals'] = st.checkbox("Pending Approvals", value=st.session_state.dashboard_widgets['Pending Approvals'], key="widget_pending_approvals")

    st.markdown("### 📊 Dashboard Insights")

    # Optional: Dashboard Insights (Conditionally displayed)
    if not df_results.empty:
        try:
            # Ensure 'Tag' column is present before trying to use it for charts
            if 'Tag' not in df_results.columns:
                df_results['Tag'] = df_results.apply(lambda row:
                    "👑 Exceptional Match" if row['Score (%)'] >= 90 and row['Years Experience'] >= 5 and row.get('Semantic Similarity', 0) >= 0.85 else (
                    "🔥 Strong Candidate" if row['Score (%)'] >= 80 and row['Years Experience'] >= 3 and row.get('Semantic Similarity', 0) >= 0.7 else (
                    "✨ Promising Fit" if row['Score (%)'] >= 60 and row['Years Experience'] >= 1 else (
                    "⚠️ Needs Review" if row['Score (%)'] >= 40 else
                    "❌ Limited Match"))), axis=1)

            col_g1, col_g2 = st.columns(2)

            if st.session_state.dashboard_widgets['Candidate Distribution']:
                with col_g1:
                    st.markdown("##### 🔥 Candidate Quality Distribution")
                    pie_data = df_results['Tag'].value_counts().reset_index()
                    pie_data.columns = ['Tag', 'Count']
                    # Use Plotly for better interactivity and dark mode handling
                    fig_plotly_pie = px.pie(pie_data, values='Count', names='Tag', title='Candidate Quality Breakdown',
                                            color_discrete_sequence=px.colors.qualitative.Pastel if not dark_mode else px.colors.qualitative.Dark2)
                    st.plotly_chart(fig_plotly_pie, use_container_width=True)

            if st.session_state.dashboard_widgets['Experience Distribution']:
                with col_g2:
                    st.markdown("##### 📊 Experience Level Breakdown")
                    bins = [0, 2, 5, 10, 20, 50] # Added 50 as upper bound for clarity
                    labels = ['0-2 yrs', '3-5 yrs', '6-10 yrs', '10-20 yrs', '20+ yrs'] # Adjusted labels
                    df_results['Experience Group'] = pd.cut(df_results['Years Experience'], bins=bins, labels=labels, right=False)
                    exp_counts = df_results['Experience Group'].value_counts().sort_index()
                    
                    # Use Plotly for better interactivity and dark mode handling
                    fig_plotly_bar = px.bar(exp_counts, x=exp_counts.index, y=exp_counts.values, title='Experience Distribution',
                                            labels={'x': 'Experience Range', 'y': 'Number of Candidates'},
                                            color_discrete_sequence=px.colors.sequential.Plasma if dark_mode else px.colors.sequential.Viridis)
                    st.plotly_chart(fig_plotly_bar, use_container_width=True)
            
            # This table is always useful, so it's not tied to a widget checkbox
            st.markdown("##### 📋 Candidate Quality Summary")
            tag_summary = df_results['Tag'].value_counts().reset_index()
            tag_summary.columns = ['Candidate Tag', 'Count']
            st.dataframe(tag_summary, use_container_width=True, hide_index=True)

            if st.session_state.dashboard_widgets['Top 5 Most Common Skills']:
                st.markdown("##### 🧠 Top 5 Matched Skills")
                if 'Matched Keywords' in df_results.columns:
                    all_skills = []
                    for skills in df_results['Matched Keywords'].dropna():
                        all_skills.extend([s.strip().lower() for s in skills.split(",") if s.strip()])
                    skill_counts = pd.Series(all_skills).value_counts().head(5)
                    if not skill_counts.empty:
                        fig_skills, ax3 = plt.subplots(figsize=(5.8, 3))
                        
                        if dark_mode:
                            palette = sns.color_palette("magma", len(skill_counts))
                        else:
                            palette = sns.color_palette("cool", len(skill_counts))
                        sns.barplot(
                            x=skill_counts.values,
                            y=skill_counts.index,
                            palette=palette,
                            ax=ax3
                        )
                        ax3.set_title("Top 5 Skills", fontsize=13, fontweight='bold', color='white' if dark_mode else 'black')
                        ax3.set_xlabel("Frequency", fontsize=11, color='white' if dark_mode else 'black')
                        ax3.set_ylabel("Skill", fontsize=11, color='white' if dark_mode else 'black')
                        ax3.tick_params(labelsize=10, colors='white' if dark_mode else 'black')
                        
                        for i, v in enumerate(skill_counts.values):
                            ax3.text(v + 0.3, i, str(v), color='white' if dark_mode else 'black', va='center', fontweight='bold', fontsize=9)
                        fig_skills.tight_layout()
                        st.pyplot(fig_skills)
                        plt.close(fig_skills)
                    else:
                        st.info("No skill data available in results for the Top 5 Skills chart.")
                else:
                    st.info("No 'Matched Keywords' column found in results for skill analysis.")
        except Exception as e:
            st.warning(f"⚠️ Could not render insights due to data error: {e}")
    
    st.markdown("---")

    # --- New Dashboard Widgets ---
    if st.session_state.dashboard_widgets['My Recent Screenings']:
        st.subheader("My Recent Screenings")
        if not df_results.empty:
            # Expander for "Resumes Screened"
            with st.expander(f"View {resume_count} Screened Resumes"):
                for idx, row in df_results.iterrows():
                    st.markdown(f"- **{row['Candidate Name']}** (Score: {row['Score (%)']:.1f}%, File: {row['File Name']})")
            st.dataframe(df_results[['Candidate Name', 'Score (%)', 'Years Experience', 'File Name']].head(5), use_container_width=True, hide_index=True)
            if st.button("View All Screenings in Analytics", key="view_all_screenings_dashboard"):
                st.session_state.tab_override = '📊 Screening Analytics'
                st.rerun()
        else:
            st.info("No recent screenings to display. Run the Resume Screener to see results here.")

    if st.session_state.dashboard_widgets['Top Performing JDs']:
        st.subheader("Top Performing Job Descriptions")
        if 'comprehensive_df' in st.session_state and not st.session_state['comprehensive_df'].empty:
            df_all_results = st.session_state['comprehensive_df'].copy()
            
            # --- START FIX: Ensure 'JD Used' column exists for display ---
            if 'JD Used' not in df_all_results.columns:
                # This is a fallback/mock for demonstration if screener.py doesn't add it
                df_all_results['JD Used'] = 'Default Job Description' 
                st.warning("Note: 'JD Used' column not found in screening results. Using 'Default Job Description' for display. Please update your screener to track the JD used.")
            # --- END FIX ---

            if 'JD Used' in df_all_results.columns: # Re-check after potential mock addition
                # Filter for shortlisted candidates based on session state criteria
                shortlisted_per_jd = df_all_results[
                    (df_all_results["Score (%)"] >= cutoff_score) &
                    (df_all_results["Years Experience"] >= min_exp_required) &
                    (df_all_results["Years Experience"] <= max_exp_allowed) & # Apply max experience filter
                    ((df_all_results['CGPA (4.0 Scale)'].isnull()) | (df_all_results['CGPA (4.0 Scale)'] >= min_cgpa_required)) # Apply CGPA filter
                ]['JD Used'].value_counts().reset_index()
                shortlisted_per_jd.columns = ['Job Description', 'Shortlisted Count']
                
                if not shortlisted_per_jd.empty:
                    st.dataframe(shortlisted_per_jd, use_container_width=True, hide_index=True)
                else:
                    st.info("No shortlisted candidates found for any JD yet based on current criteria.")
            else:
                st.info("Still unable to determine top performing JDs. 'JD Used' column is missing even after fallback.")
        else:
            st.info("No screening results available to determine top performing JDs.")
        
        if st.button("Manage All Job Descriptions", key="manage_all_jds_dashboard"):
            st.session_state.tab_override = '📁 Manage JDs'
            st.rerun()

    if st.session_state.dashboard_widgets['Pending Approvals']:
        st.subheader("Pending Approvals")
        if 'pending_approvals' not in st.session_state:
            st.session_state.pending_approvals = []

        if not st.session_state.pending_approvals:
            st.info("No candidates currently awaiting approval.")
            # Add a mock button to add a candidate for approval for testing
            if st.button("Mock: Add Candidate for Approval"):
                mock_candidate = {
                    "candidate_name": f"Mock Candidate {len(st.session_state.pending_approvals) + 1}",
                    "score": 85,
                    "experience": 4,
                    "jd_used": "Business Analyst",
                    "status": "pending",
                    "notes": "Good potential, needs managerial review."
                }
                st.session_state.pending_approvals.append(mock_candidate)
                log_activity(f"Mock candidate '{mock_candidate['candidate_name']}' added for approval.")
                st.rerun()
        else:
            st.write("Review the following candidates:")
            for i, candidate in enumerate(st.session_state.pending_approvals):
                if candidate['status'] == 'pending':
                    with st.expander(f"Candidate: {candidate['candidate_name']} (Score: {candidate['score']}%)"):
                        st.write(f"**JD Used:** {candidate['jd_used']}")
                        st.write(f"**Experience:** {candidate['experience']} years")
                        st.write(f"**Notes:** {candidate['notes']}")
                        
                        col_approve, col_reject = st.columns(2)
                        with col_approve:
                            if st.button(f"✅ Approve {candidate['candidate_name']}", key=f"approve_{i}"):
                                st.session_state.pending_approvals[i]['status'] = 'approved'
                                log_activity(f"Candidate '{candidate['candidate_name']}' approved.")
                                st.success(f"Approved {candidate['candidate_name']}!")
                                st.rerun()
                        with col_reject:
                            if st.button(f"❌ Reject {candidate['candidate_name']}", key=f"reject_{i}"):
                                st.session_state.pending_approvals[i]['status'] = 'rejected'
                                log_activity(f"Candidate '{candidate['candidate_name']}' rejected.")
                                st.error(f"Rejected {candidate['candidate_name']}.")
                                st.rerun()
            # Optionally show approved/rejected candidates
            approved_rejected = [c for c in st.session_state.pending_approvals if c['status'] != 'pending']
            if approved_rejected:
                st.markdown("---")
                st.subheader("Reviewed Candidates")
                reviewed_df = pd.DataFrame(approved_rejected)
                st.dataframe(reviewed_df[['candidate_name', 'score', 'experience', 'status']], use_container_width=True, hide_index=True)


    # Removed Activity Feed as per user request
    # if st.session_state.dashboard_widgets['Activity Feed']:
    #     st.subheader("Recent Activity Feed")
    #     if st.session_state.get('activity_log'):
    #         # Display only the last 10 activities for brevity on dashboard
    #         for activity in st.session_state.activity_log[:10]:
    #             st.markdown(f"- <small>{activity}</small>", unsafe_allow_html=True)
    #         if len(st.session_state.activity_log) > 10:
    #             st.info(f"Showing last 10 activities. Total activities: {len(st.session_state.activity_log)}. Scroll up to customize widget display.")
    #     else:
    #         st.info("No recent activity.")


# ======================
# ⚙️ Admin Tools Section
# ======================
elif tab == "⚙️ Admin Tools":
    st.markdown('<div class="dashboard-header">⚙️ Admin Tools</div>', unsafe_allow_html=True)
    if is_admin:
        st.write("Welcome, Administrator! Here you can manage user accounts.")
        st.markdown("---")
        admin_registration_section() # Create New User Form
        st.markdown("---")
        admin_password_reset_section() # Reset User Password Form
        st.markdown("---")
        admin_disable_enable_user_section() # Disable/Enable User Form
        st.markdown("---")
        st.subheader("👥 All Registered Users")
        st.warning("⚠️ **SECURITY WARNING:** This table displays usernames (email IDs) and **hashed passwords**. This is for **ADMINISTRATIVE DEBUGGING ONLY IN A SECURE ENVIRONMENT**. **NEVER expose this in a public or production application.**")
        try:
            users_data = load_users()
            if users_data:
                display_users = []
                for user, data in users_data.items():
                    hashed_pass = data.get("password", data) if isinstance(data, dict) else data
                    status = data.get("status", "N/A") if isinstance(data, dict) else "N/A"
                    display_users.append([user, hashed_pass, status])
                st.dataframe(pd.DataFrame(display_users, columns=["Email/Username", "Hashed Password (DO NOT EXPOSE)", "Status"]), use_container_width=True)
            else:
                st.info("No users registered yet.")
        except Exception as e:
            st.error(f"Error loading user data: {e}")
    else:
        st.error("🔒 Access Denied: You must be an administrator to view this page.")

# ======================
# Page Routing via function calls (remaining pages)
# ======================
# Placeholder for screener.py, email_sender.py, analytics.py, manage_jds.py, search.py, notes.py
# You need to ensure these files exist and define the respective page functions.
# For demonstration, these will print a message if the files are not available.

elif tab == "🧠 Resume Screener":
    try:
        from screener import resume_screener_page
        resume_screener_page()
        # Example of logging an activity after screening (you'd integrate this inside screener.py)
        if 'comprehensive_df' in st.session_state and not st.session_state['comprehensive_df'].empty:
            # This logic should ideally be inside screener.py when a new screening is completed
            # For demonstration, we'll log if new results appear
            if st.session_state.get('last_screen_log_count', 0) < len(st.session_state['comprehensive_df']):
                log_activity(f"Performed resume screening for {len(st.session_state['comprehensive_df'])} candidates.")
                st.session_state.last_screen_log_count = len(st.session_state['comprehensive_df'])

            # Example: Triggering a pending approval for a high-scoring candidate
            # This would be more sophisticated in screener.py, checking specific criteria
            for result in st.session_state['comprehensive_df'].to_dict('records'): # Iterate over records
                if result['Score (%)'] >= 90 and result['Candidate Name'] not in [app['candidate_name'] for app in st.session_state.get('pending_approvals', []) if app['status'] == 'pending']:
                    if 'pending_approvals' not in st.session_state:
                        st.session_state.pending_approvals = []
                    st.session_state.pending_approvals.append({
                        "candidate_name": result['Candidate Name'],
                        "score": result['Score (%)'],
                        "experience": result['Years Experience'],
                        "jd_used": result.get('JD Used', 'N/A'), # Ensure JD Used is captured
                        "status": "pending",
                        "notes": f"High-scoring candidate from recent screening."
                    })
                    log_activity(f"Candidate '{result['Candidate Name']}' sent for approval (high score).")
                    st.toast(f"Candidate {result['Candidate Name']} sent for approval!")


    except ImportError:
        st.info("`screener.py` not found or function not defined. Please create it.")
    except Exception as e:
        st.error(f"Error loading Resume Screener: {e}")

elif tab == "📁 Manage JDs":
    try:
        # Assuming manage_jds.py contains its Streamlit code directly or in a function
        with open("manage_jds.py", encoding="utf-8") as f:
            exec(f.read())
        # Example of logging an activity after JD update (you'd integrate this inside manage_jds.py)
        # log_activity("Job Description updated/added.")
    except FileNotFoundError:
        st.info("`manage_jds.py` not found. Please ensure the file exists in the same directory.")
    except Exception as e:
        st.error(f"Error loading Manage JDs: {e}")

elif tab == "📊 Screening Analytics":
    try:
        from analytics import analytics_dashboard_page
        analytics_dashboard_page()
    except ImportError:
        st.info("`analytics.py` not found or function not defined. Please create it.")
    except Exception as e:
        st.error(f"Error loading Screening Analytics: {e}")

elif tab == "📤 Email Candidates":
    try:
        from email_sender import send_email_to_candidate
        send_email_to_candidate()
        # Example of logging an activity after sending email
        # log_activity("Emails sent to candidates.")
    except ImportError:
        st.info("`email_sender.py` not found or function not defined. Please create it.")
    except Exception as e:
        st.error(f"Error loading Email Candidates: {e}")

elif tab == "🔍 Search Resumes":
    try:
        # Assuming search.py contains its Streamlit code directly or in a function
        with open("search.py", encoding="utf-8") as f:
            exec(f.read())
        # Example of logging an activity after searching
        # log_activity("Performed resume search.")
    except FileNotFoundError:
        st.info("`search.py` not found. Please ensure the file exists in the same directory.")
    except Exception as e:
        st.error(f"Error loading Search Resumes: {e}")

elif tab == "📝 Candidate Notes":
    try:
        # Assuming notes.py contains its Streamlit code directly or in a function
        with open("notes.py", encoding="utf-8") as f:
            exec(f.read())
        # Example of logging an activity after adding note
        # log_activity("Added candidate note.")
    except FileNotFoundError:
        st.info("`notes.py` not found. Please ensure the file exists in the same directory.")
    except Exception as e:
        st.error(f"Error loading Candidate Notes: {e}")

# NEW: Feedback & Help section
elif tab == "❓ Feedback & Help":
    # Ensure a user_email is set for logging purposes
    if 'user_email' not in st.session_state:
        st.session_state['user_email'] = st.session_state.get('username', 'anonymous_user')
    feedback_and_help_page() # Call the feedback page function

elif tab == "🚪 Logout":
    log_activity(f"User '{st.session_state.get('username', 'anonymous_user')}' logged out.")
    st.session_state.authenticated = False
    st.session_state.pop('username', None)
    st.success("✅ Logged out.")
    st.rerun() # Rerun to redirect to login page
