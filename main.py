import streamlit as st
import pandas as pd
import os
from datetime import datetime
import collections
import re # Still needed for general text processing if any in main.py, otherwise remove
import base64
from io import BytesIO

# Import authentication functions from the new login.py file
# Ensure these imports match the functions available in your login.py
from login import (
    load_users, save_users, hash_password, check_password, is_valid_email,
    admin_registration_section, admin_password_reset_section, admin_disable_enable_user_section,
    login_section_ui, is_current_user_admin, is_current_user_hr, is_current_user_candidate,
    ADMIN_USERNAME # Import ADMIN_USERNAME for initial setup
)

# Import the new pages (ensure these files exist in your directory)
from about_us import about_us_page
from privacy_policy import privacy_policy_page
# Import the resume_screener_page from screener.py
from screener import resume_screener_page

# --- Define load_css function ---
def load_css(css_file_name):
    """Loads a CSS file and applies it to the Streamlit app."""
    try:
        with open(css_file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{css_file_name}' not found. Default styles will be used.")
    except Exception as e:
        st.error(f"Error loading CSS file '{css_file_name}': {e}")

# --- Firebase REST Setup ---
# These should ideally be in a secrets.toml file or environment variables in production
FIREBASE_WEB_API_KEY = os.environ.get('FIREBASE_WEB_API_KEY', 'AIzaSyDjC7tdmpEkpsipgf9r1c3HlTO7C7BZ6Mw')
FIREBASE_PROJECT_ID = globals().get('__app_id', 'screenerproapp') # Use __app_id for Canvas environment
FIRESTORE_BASE_URL = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents"

# Helper for Activity Logging (for main.py's own activities)
def log_activity_main(message):
    if 'activity_log' not in st.session_state:
        st.session_state.activity_log = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.activity_log.insert(0, f"[{timestamp}] {message}")
    st.session_state.activity_log = st.session_state.activity_log[:50]

# --- Firebase Data Persistence Functions (REST API) ---
# These functions would typically be in a separate file (e.g., firebase_utils.py)
# and imported. For this consolidated example, they are defined here.

# Function to load session data (e.g., user details) from Firestore
def load_session_data_from_firestore_rest(username):
    """Loads user session data from Firestore for a given username."""
    try:
        # Construct the URL for the user's document
        user_doc_path = f"users/{username}"
        url = f"{FIRESTORE_BASE_URL}/{user_doc_path}?key={FIREBASE_WEB_API_KEY}"
        
        # Make a GET request to Firestore
        response = requests.get(url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        
        data = response.json()
        
        # Firestore returns a document object, extract fields
        if 'fields' in data:
            user_data = {}
            for key, value_obj in data['fields'].items():
                # Extract the actual value based on its type
                if 'stringValue' in value_obj:
                    user_data[key] = value_obj['stringValue']
                elif 'integerValue' in value_obj:
                    user_data[key] = int(value_obj['integerValue'])
                elif 'booleanValue' in value_obj:
                    user_data[key] = value_obj['booleanValue']
                elif 'arrayValue' in value_obj and 'values' in value_obj['arrayValue']:
                    user_data[key] = [v['stringValue'] for v in value_obj['arrayValue']['values'] if 'stringValue' in v]
                # Add other types as needed (e.g., mapValue, doubleValue)
            return user_data
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Firestore load error: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during Firestore load: {e}")
        return None

# Function to save session data (e.g., user details) to Firestore
def save_session_data_to_firestore_rest(username, data):
    """Saves user session data to Firestore for a given username."""
    try:
        user_doc_path = f"users/{username}"
        url = f"{FIRESTORE_BASE_URL}/{user_doc_path}?key={FIREBASE_WEB_API_KEY}"
        
        # Convert Python dict to Firestore's expected JSON format
        firestore_data = {"fields": {}}
        for key, value in data.items():
            if isinstance(value, str):
                firestore_data["fields"][key] = {"stringValue": value}
            elif isinstance(value, int):
                firestore_data["fields"][key] = {"integerValue": str(value)} # Firestore expects string for integerValue
            elif isinstance(value, bool):
                firestore_data["fields"][key] = {"booleanValue": value}
            elif isinstance(value, list):
                firestore_data["fields"][key] = {"arrayValue": {"values": [{"stringValue": v} for v in value]}}
            # Add other types as needed
        
        # Use PATCH for updating existing document, PUT for creating/overwriting
        response = requests.patch(url, json=firestore_data)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Firestore save error: {e}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during Firestore save: {e}")
        return False

# Placeholder for analytics_dashboard_page
def analytics_dashboard_page():
    st.subheader("📊 Screening Analytics Dashboard")
    st.write("This page will display comprehensive screening analytics and insights.")
    st.info("Feature under development. Stay tuned for advanced dashboards!")

# --- Streamlit App Layout ---
st.set_page_config(page_title="ScreenerPro – AI Hiring Dashboard", layout="wide", page_icon="🧠")

load_css("style.css") # Load CSS

dark_mode = st.sidebar.toggle("🌙 Dark Mode", key="dark_mode_main_init")

# Apply dark mode styles initially
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {'#1E1E1E' if dark_mode else '#F0F2F6'};
    color: {'#E0E0E0' if dark_mode else '#333333'};
}}
.main .block-container {{
    background: {'#2D2D2D' if dark_mode else 'rgba(255, 255, 255, 0.96)'};
}}
</style>
""", unsafe_allow_html=True)


st.sidebar.image("logo.png", width=200)
st.sidebar.title("🧠 ScreenerPro")

# --- Initialize Session State for Portal Selection ---
if 'active_portal' not in st.session_state:
    st.session_state.active_portal = None # Start with no portal selected
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None

# --- Top-level Portal Selection Buttons (only shown if not authenticated) ---
if not st.session_state.authenticated:
    st.markdown("## Choose Your Portal")
    col_hr_portal, col_candidate_portal = st.columns(2)

    with col_hr_portal:
        if st.button("🔹 HR Portal", use_container_width=True, help="Access for HR personnel and Administrators", key="btn_hr_portal_main"):
            st.session_state.active_portal = "hr_portal"
            if f"active_login_tab_selection_candidate_portal" in st.session_state:
                del st.session_state[f"active_login_tab_selection_candidate_portal"]
            st.rerun()

    with col_candidate_portal:
        if st.button("🔸 Candidate Portal", use_container_width=True, help="Access for job candidates", key="btn_candidate_portal_main"):
            st.session_state.active_portal = "candidate_portal"
            if f"active_login_tab_selection_hr_portal" in st.session_state:
                del st.session_state[f"active_login_tab_selection_hr_portal"]
            st.rerun()

    st.markdown("---") # Separator

    # Display login/register UI based on selected portal
    if st.session_state.active_portal == "hr_portal":
        st.header("HR Portal")
        authenticated_for_portal = login_section_ui("hr_portal", load_session_data_from_firestore_rest)
        if not authenticated_for_portal:
            st.stop()
    elif st.session_state.active_portal == "candidate_portal":
        st.header("Candidate Portal")
        authenticated_for_portal = login_section_ui("candidate_portal")
        if not authenticated_for_portal:
            st.stop()
    else:
        st.info("Please select a portal to log in or register.")
        st.stop()
    
# --- Main Application Content (only runs if authenticated) ---
if st.session_state.authenticated:
    if st.session_state.get('last_login_logged_for_user') != st.session_state.username:
        log_activity_main(f"User '{st.session_state.username}' logged in to {st.session_state.user_role.upper()} Portal.")
        st.session_state.last_login_logged_for_user = st.session_state.username

    st.sidebar.success(f"Logged in as {st.session_state.username}")
    st.sidebar.info(f"Company/Name: {st.session_state.user_company}")
    st.sidebar.info(f"Role: {st.session_state.user_role.upper()}")

    st.markdown(f'<div class="greeting-message">Hello, {st.session_state.username}! 👋</div>', unsafe_allow_html=True)

    if st.session_state.user_role == "hr" or st.session_state.user_role == "admin":
        if 'comprehensive_df' not in st.session_state:
            st.session_state['comprehensive_df'] = pd.DataFrame()

    if is_current_user_hr():
        st.header("HR Portal")
        navigation_options = [
            "🏠 Dashboard", "🧠 Resume Screener", "📁 Manage JDs", "📊 Screening Analytics",
            "📦 Bulk Resume Import", 
            "📤 Email Candidates", "🔍 Search Resumes", "📝 Candidate Notes",
            "📈 Advanced Tools",
            "🤝 Collaboration Hub",
            "🏢 About Us",
            "⚖️ Privacy Policy & Terms",
            "❓ Feedback & Help"
        ]
        if is_current_user_admin():
            navigation_options.append("⚙️ Admin Tools")
        navigation_options.append("🚪 Logout")

        default_tab = st.session_state.get("tab_override", "🏠 Dashboard")
        if default_tab not in navigation_options:
            default_tab = "🏠 Dashboard"
        tab = st.sidebar.radio("📍 Navigate (HR Portal)", navigation_options, index=navigation_options.index(default_tab), key="hr_portal_sidebar_radio_main")

    elif is_current_user_candidate():
        st.header("Candidate Portal")
        navigation_options = [
            "🧠 Resume Screener",
            "🏢 About Us",
            "⚖️ Privacy Policy & Terms",
            "❓ Feedback & Help",
            "🚪 Logout"
        ]
        default_tab = st.session_state.get("tab_override", "🧠 Resume Screener")
        if default_tab not in navigation_options:
            default_tab = "🧠 Resume Screener"
        tab = st.sidebar.radio("📍 Navigate (Candidate Portal)", navigation_options, index=navigation_options.index(default_tab), key="candidate_portal_sidebar_radio_main")
    else:
        st.error("Unknown user role. Please contact support.")
        st.session_state.authenticated = False
        st.session_state.pop('username', None)
        st.session_state.pop('user_company', None)
        st.session_state.pop('user_role', None)
        st.session_state.pop('active_portal', None)
        st.rerun()

    if "tab_override" in st.session_state:
        del st.session_state.tab_override

    if tab == "🏠 Dashboard":
        if is_current_user_hr():
            analytics_dashboard_page()
        else:
            st.error("Access Denied: You do not have permission to view the Dashboard.")

    elif tab == "🧠 Resume Screener":
        resume_screener_page() # Call the imported resume screener page

    elif tab == "📁 Manage JDs":
        if is_current_user_hr():
            try:
                with open("manage_jds.py", encoding="utf-8") as f:
                    exec(f.read())
            except FileNotFoundError:
                st.info("`manage_jds.py` not found. Please ensure the file exists in the same directory.")
            except Exception as e:
                st.error(f"Error loading Manage JDs: {e}")
        else:
            st.error("Access Denied: You do not have permission to manage JDs.")

    elif tab == "📊 Screening Analytics":
        if is_current_user_hr():
            analytics_dashboard_page()
        else:
            st.error("Access Denied: You do not have permission to view Screening Analytics.")

    elif tab == "📈 Advanced Tools":
        if is_current_user_hr():
            from advanced import advanced_tools_page
            advanced_tools_page(
                app_id=FIREBASE_PROJECT_ID,
                FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY,
                FIRESTORE_BASE_URL=FIRESTORE_BASE_URL
            )
        else:
            st.error("Access Denied: You do not have permission to access Advanced Tools.")

    elif tab == "🤝 Collaboration Hub":
        if is_current_user_hr():
            from collaboration import collaboration_hub_page
            collaboration_hub_page(
                app_id=FIREBASE_PROJECT_ID,
                FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY,
                FIRESTORE_BASE_URL=FIRESTORE_BASE_URL
            )
        else:
            st.error("Access Denied: You do not have permission to access the Collaboration Hub.")

    elif tab == "🏢 About Us":
        about_us_page()

    elif tab == "⚖️ Privacy Policy & Terms":
        privacy_policy_page()

    elif tab == "📤 Email Candidates":
        if is_current_user_hr():
            try:
                from email_sender import send_email_to_candidate
                send_email_to_candidate()
            except ImportError:
                st.error("`email_sender.py` not found or `send_email_to_candidate` function not defined. Please ensure 'email_sender.py' exists and contains the 'send_email_to_candidate' function.")
            except Exception as e:
                st.error(f"Error loading Email Candidates: {e}")
        else:
            st.error("Access Denied: You do not have permission to email candidates.")

    elif tab == "🔍 Search Resumes":
        if is_current_user_hr():
            try:
                with open("search.py", encoding="utf-8") as f:
                    exec(f.read())
            except FileNotFoundError:
                st.info("`search.py` not found. Please ensure the file exists in the same directory.")
            except Exception as e:
                st.error(f"Error loading Search Resumes: {e}")
        else:
            st.error("Access Denied: You do not have permission to search resumes.")

    elif tab == "📝 Candidate Notes":
        if is_current_user_hr():
            try:
                with open("notes.py", encoding="utf-8") as f:
                    exec(f.read())
            except FileNotFoundError:
                st.info("`notes.py` not found. Please ensure the file exists in the same directory.")
            except Exception as e:
                st.error(f"Error loading Candidate Notes: {e}")
        else:
            st.error("Access Denied: You do not have permission to view candidate notes.")

    elif tab == "❓ Feedback & Help":
        from feedback import feedback_and_help_page
        if 'user_email' not in st.session_state:
            st.session_state['user_email'] = st.session_state.get('username', 'anonymous_user')
        feedback_and_help_page()

    elif tab == "📦 Bulk Resume Import":
        if is_current_user_hr():
            from bulk_upload_page import bulk_upload_page
            jd_texts = {}
            if os.path.exists("data"):
                for fname in os.listdir("data"):
                    if fname.endswith(".txt"):
                        with open(os.path.join("data", fname), "r", encoding="utf-8") as f:
                            jd_texts[fname.replace(".txt", "").replace("_", " ").title()] = f.read()
            bulk_upload_page(st.session_state['comprehensive_df'], jd_texts)
        else:
            st.error("Access Denied: You do not have permission for Bulk Resume Import.")

    elif tab == "⚙️ Admin Tools":
        st.markdown('<div class="dashboard-header">⚙️ Admin Tools</div>', unsafe_allow_html=True)
        if is_current_user_admin():
            st.subheader(f"Welcome, Admin {st.session_state.username}!")
            admin_tab = st.tabs(["Create User", "Reset Password", "Toggle User Status", "Activity Log"])

            with admin_tab[0]:
                admin_registration_section()
            with admin_tab[1]:
                admin_password_reset_section()
            with admin_tab[2]:
                admin_disable_enable_user_section()
            with admin_tab[3]:
                st.subheader("Recent Activity Log (Main App)")
                if 'activity_log' in st.session_state and st.session_state.activity_log:
                    for log_entry in st.session_state.activity_log:
                        st.text(log_entry)
                else:
                    st.info("No recent activity to display.")
        else:
            st.error("Access Denied: You do not have administrative privileges to view this page.")

    elif tab == "🚪 Logout":
        log_activity_main(f"User '{st.session_state.get('username', 'anonymous_user')}' logged out.")
        st.session_state.authenticated = False
        st.session_state.pop('username', None)
        st.session_state.pop('user_company', None)
        st.session_state.pop('user_role', None)
        st.session_state.pop('active_portal', None)
        st.success("✅ Logged out.")
        st.rerun()
