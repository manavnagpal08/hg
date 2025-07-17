import streamlit as st
import json
import bcrypt
import os
import re
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import statsmodels.api as sm
import collections
import requests

# Import the new pages
from about_us import about_us_page
from privacy_policy import privacy_policy_page
# Import the screener functions from the updated screener.py
from screener import resume_screener_page, candidate_screener_page

# --- Firebase REST Setup (for main.py's own data persistence, e.g., session data) ---
# IMPORTANT: Replace "YOUR_FIREBASE_WEB_API_KEY" with your actual Firebase Web API Key
# You can find this in your Firebase project settings -> Project settings -> General -> Web API Key
# For Canvas, __app_id is provided. For local dev, use a default.
FIREBASE_WEB_API_KEY = os.environ.get('FIREBASE_WEB_API_KEY', st.secrets.get("FIREBASE_WEB_API_KEY", "YOUR_FIREBASE_WEB_API_KEY_HERE"))
FIREBASE_PROJECT_ID = globals().get('__app_id', st.secrets.get("FIREBASE_PROJECT_ID", 'screenerpro-default-app-id'))
FIRESTORE_BASE_URL = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents"

# File to store user credentials (for HR users)
USER_DB_FILE = "users.json"
# Define your admin usernames here as a tuple of strings
ADMIN_USERNAME = ("admin@forscreenerpro", "admin@forscreenerpro2", "manav.nagpal2005@gmail.com")

def load_users():
    """Loads user data from the JSON file."""
    if not os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "w") as f:
            json.dump({}, f)
    with open(USER_DB_FILE, "r") as f:
        users = json.load(f)
        # Ensure each user has a 'status' key and 'company' key for backward compatibility
        for username, data in users.items():
            if isinstance(data, str): # Old format: "username": "hashed_password"
                users[username] = {"password": data, "status": "active", "company": "N/A"}
            # After ensuring it's a dict, check for missing keys
            if "status" not in users[username]:
                users[username]["status"] = "active"
            if "company" not in users[username]: # Add company field if missing
                users[username]["company"] = "N/A"
        return users

def save_users(users):
    """Saves user data to the JSON file."""
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f, indent=4)

def hash_password(password):
    """Hashes a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed_password):
    """Checks a password against its bcrypt hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def is_valid_email(email):
    """Basic validation for email format."""
    return re.match(r"[^@]+@[^@]+\.\w+", email)

def register_section():
    """Public self-registration form."""
    st.subheader("📝 Create New Account")
    with st.form("registration_form", clear_on_submit=True):
        new_username = st.text_input("Choose Username (Email address required)", key="new_username_reg_public")
        new_company_name = st.text_input("Company Name", key="new_company_name_reg_public")
        new_password = st.text_input("Choose Password", type="password", key="new_password_reg_public")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password_reg_public")
        register_button = st.form_submit_button("Register New Account")

        if register_button:
            if not new_username or not new_password or not confirm_password or not new_company_name:
                st.error("Please fill in all fields.")
            elif not is_valid_email(new_username):
                st.error("Please enter a valid email address for the username.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                users = load_users()
                if new_username in users:
                    st.error("Username already exists. Please choose a different one.")
                else:
                    users[new_username] = {
                        "password": hash_password(new_password),
                        "status": "active",
                        "company": new_company_name
                    }
                    save_users(users)
                    st.success("✅ Registration successful! You can now switch to the 'Login' option.")
                    st.session_state.active_login_tab_selection = "Login"

def admin_registration_section():
    """Admin-driven user creation form."""
    st.subheader("➕ Create New User Account (Admin Only)")
    with st.form("admin_registration_form", clear_on_submit=True):
        new_username = st.text_input("New User's Username (Email)", key="new_username_admin_reg")
        new_company_name = st.text_input("New User's Company Name", key="new_company_name_admin_reg")
        new_password = st.text_input("New User's Password", type="password", key="new_password_admin_reg")
        admin_register_button = st.form_submit_button("Add New User")

    if admin_register_button:
        if not new_username or not new_password or not new_company_name:
            st.error("Please fill in all fields.")
        elif not is_valid_email(new_username):
            st.error("Please enter a valid email address for the username.")
        else:
            users = load_users()
            if new_username in users:
                st.error(f"User '{new_username}' already exists.")
            else:
                users[new_username] = {
                    "password": hash_password(new_password),
                    "status": "active",
                    "company": new_company_name
                }
                save_users(users)
                st.success(f"✅ User '{new_username}' added successfully!")

def admin_password_reset_section():
    """Admin-driven password reset form."""
    st.subheader("🔑 Reset User Password (Admin Only)")
    users = load_users()
    user_options = [user for user in users.keys() if user not in ADMIN_USERNAME]

    if not user_options:
        st.info("No other users to reset passwords for.")
        return

    with st.form("admin_reset_password_form", clear_on_submit=True):
        selected_user = st.selectbox("Select User to Reset Password For", user_options, key="reset_user_select")
        new_password = st.text_input("New Password", type="password", key="new_pwd_reset")
        reset_button = st.form_submit_button("Reset Password")

        if reset_button:
            if not new_password:
                st.error("Please enter a new password.")
            else:
                users[selected_user]["password"] = hash_password(new_password)
                save_users(users)
                st.success(f"✅ Password for '{selected_user}' has been reset.")

def admin_disable_enable_user_section():
    """Admin-driven user disable/enable form."""
    st.subheader("⛔ Toggle User Status (Admin Only)")
    users = load_users()
    user_options = [user for user in users.keys() if user not in ADMIN_USERNAME]

    if not user_options:
        st.info("No other users to manage status for.")
        return

    with st.form("admin_toggle_user_status_form", clear_on_submit=False):
        selected_user = st.selectbox("Select User to Toggle Status", user_options, key="toggle_user_select")

        current_status = users[selected_user]["status"]
        st.info(f"Current status of '{selected_user}': **{current_status.upper()}**")

        if st.form_submit_button(f"Toggle to {'Disable' if current_status == 'active' else 'Enable'} User"):
            new_status = "disabled" if current_status == "active" else "active"
            users[selected_user]["status"] = new_status
            save_users(users)
            st.success(f"✅ User '{selected_user}' status set to **{new_status.upper()}**.")
            st.rerun()

# --- Firebase Data Persistence Functions (REST API) ---
def to_firestore_format(data: dict) -> dict:
    """Converts a Python dictionary to Firestore REST API 'fields' format."""
    fields = {}
    for key, value in data.items():
        if isinstance(value, str):
            fields[key] = {"stringValue": value}
        elif isinstance(value, int):
            fields[key] = {"integerValue": str(value)}
        elif isinstance(value, float):
            fields[key] = {"doubleValue": value}
        elif isinstance(value, bool):
            fields[key] = {"booleanValue": value}
        elif isinstance(value, datetime):
            fields[key] = {"timestampValue": value.isoformat() + "Z"}
        elif isinstance(value, list):
            array_values = []
            for item in value:
                if isinstance(item, str):
                    array_values.append({"stringValue": item})
                elif isinstance(item, int):
                    array_values.append({"integerValue": str(item)})
                elif isinstance(item, float):
                    array_values.append({"doubleValue": item})
                elif isinstance(item, bool):
                    array_values.append({"booleanValue": item})
                elif isinstance(item, dict):
                    array_values.append({"mapValue": {"fields": to_firestore_format(item)['fields']}})
            fields[key] = {"arrayValue": {"values": array_values}}
        elif isinstance(value, dict):
            fields[key] = {"mapValue": {"fields": to_firestore_format(value)['fields']}}
        elif value is None:
            fields[key] = {"nullValue": None}
        else:
            fields[key] = {"stringValue": str(value)}
    return {"fields": fields}


def save_session_data_to_firestore_rest(username):
    """
    Saves comprehensive session data (including comprehensive_df) to Firestore
    using the REST API for the current user.
    """
    try:
        if not username:
            st.warning("No username found. Please log in.")
            return

        doc_path = f"artifacts/{FIREBASE_PROJECT_ID}/users/{username}/session_data_rest/current_session"
        url = f"{FIRESTORE_BASE_URL}/{doc_path}?key={FIREBASE_WEB_API_KEY}"

        data_to_save = {}
        if 'comprehensive_df' in st.session_state and not st.session_state['comprehensive_df'].empty:
            df_for_save = st.session_state['comprehensive_df'].copy()

            if 'Shortlisted' not in df_for_save.columns:
                shortlist_threshold = st.session_state.get('screening_cutoff_score', 75)
                df_for_save['Shortlisted'] = df_for_save['Score (%)'].apply(
                    lambda x: f"Yes (Score >= {shortlist_threshold}%)" if x >= shortlist_threshold else "No"
                )
                log_activity_main("Derived 'Shortlisted' column for saving as it was missing.")

            df_for_save = df_for_save.drop(columns=['Resume Raw Text'], errors='ignore')
            data_to_save['comprehensive_df_json'] = df_for_save.to_json(orient='records')

            shortlisted_candidates = df_for_save[df_for_save['Shortlisted'].str.startswith('Yes')]['Candidate Name'].tolist()
            job_locations = df_for_save['Location'].dropna().unique().tolist()

            data_to_save['shortlisted_names'] = json.dumps(shortlisted_candidates)
            data_to_save['job_locations_found'] = json.dumps(job_locations)
            data_to_save['screened_count'] = len(df_for_save)
            data_to_save['timestamp'] = str(datetime.now())
            data_to_save['status'] = "saved from Streamlit Cloud"

        else:
            st.warning("No comprehensive data to save.")
            return

        firestore_data = to_firestore_format(data_to_save)

        res = requests.patch(url, json=firestore_data)
        if res.status_code in [200, 201]:
            st.success("✅ Session data saved to Cloud (REST API)!")
            log_activity_main(f"Session data saved for user '{username}' via REST API.")
        else:
            st.error(f"❌ REST Save failed: {res.status_code}, {res.text}")
            log_activity_main(f"REST Save failed for user '{username}': {res.status_code}, {res.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"🔥 REST Firebase connection error: {e}")
        log_activity_main(f"REST Firebase connection error for user '{username}': {e}")
    except Exception as e:
        st.error(f"🔥 An unexpected error occurred during REST save: {e}")
        log_activity_main(f"Unexpected REST save error for user '{username}': {e}")

def from_firestore_format(firestore_data: dict) -> dict:
    """Converts Firestore REST API 'fields' format to a Python dictionary."""
    data = {}
    if "fields" not in firestore_data:
        return data
    
    for key, value_obj in firestore_data["fields"].items():
        if "stringValue" in value_obj:
            data[key] = value_obj["stringValue"]
        elif "integerValue" in value_obj:
            data[key] = int(value_obj["integerValue"])
        elif "doubleValue" in value_obj:
            data[key] = float(value_obj["doubleValue"])
        elif "booleanValue" in value_obj:
            data[key] = value_obj["booleanValue"]
        elif "timestampValue" in value_obj:
            try:
                data[key] = datetime.fromisoformat(value_obj["timestampValue"].replace('Z', ''))
            except ValueError:
                data[key] = value_obj["timestampValue"]
        elif "arrayValue" in value_obj and "values" in value_obj["arrayValue"]:
            data[key] = [from_firestore_format({"fields": {"_": item}})["_"] if "mapValue" not in item else from_firestore_format({"fields": item["mapValue"]["fields"]}) for item in value_obj["arrayValue"]["values"]]
        elif "mapValue" in value_obj and "fields" in value_obj["mapValue"]:
            data[key] = from_firestore_format({"fields": value_obj["mapValue"]["fields"]})
        elif "nullValue" in value_obj:
            data[key] = None
    return data


def load_session_data_from_firestore_rest(username):
    """
    Loads comprehensive session data from Firestore using the REST API
    for the current user.
    """
    try:
        if not username:
            st.warning("No username found. Please log in.")
            return

        doc_path = f"artifacts/{FIREBASE_PROJECT_ID}/users/{username}/session_data_rest/current_session"
        url = f"{FIRESTORE_BASE_URL}/{doc_path}?key={FIREBASE_WEB_API_KEY}"

        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            loaded_data = from_firestore_format(data)

            if 'comprehensive_df_json' in loaded_data:
                df_json_content = loaded_data['comprehensive_df_json']
                try:
                    if isinstance(df_json_content, str):
                        st.session_state['comprehensive_df'] = pd.read_json(df_json_content, orient='records')
                    else:
                        st.session_state['comprehensive_df'] = pd.DataFrame.from_records(df_json_content)

                    st.toast("Session data loaded from Cloud (REST API)!")
                    log_activity_main(f"Session data loaded for user '{username}' via REST API.")
                except Exception as e:
                    st.error(f"Error reconstructing DataFrame from loaded JSON: {e}. Data might be corrupted or incompatible. Raw JSON content (truncated): {str(df_json_content)[:200]}...")
                    log_activity_main(f"Error reconstructing DataFrame for user '{username}': {e}")
                    st.session_state['comprehensive_df'] = pd.DataFrame()
            else:
                st.info("No comprehensive data found in the loaded session data.")
                st.session_state['comprehensive_df'] = pd.DataFrame()
        elif res.status_code == 404:
            st.info("No previous session data found in Cloud (REST API).")
            st.session_state['comprehensive_df'] = pd.DataFrame()
        else:
            st.error(f"❌ REST Load failed: {res.status_code}, {res.text}")
            log_activity_main(f"REST Load failed for user '{username}': {res.status_code}, {res.text}")
            st.session_state['comprehensive_df'] = pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"🔥 REST Firebase connection error: {e}")
        log_activity_main(f"REST Firebase connection error for user '{username}': {e}")
        st.session_state['comprehensive_df'] = pd.DataFrame()
    except Exception as e:
        st.error(f"🔥 An unexpected error occurred during REST load: {e}")
        log_activity_main(f"Unexpected REST load error for user '{username}': {e}")
        st.session_state['comprehensive_df'] = pd.DataFrame()


def login_section():
    """
    Handles user login (HR) and public registration, and mode selection.
    Sets session state variables for authentication and user details.
    Returns True if authenticated (HR) or logged in (Candidate), False otherwise.
    """
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "user_company" not in st.session_state:
        st.session_state.user_company = None
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = "select_mode" # Default mode before selection
    if 'candidate_logged_in' not in st.session_state:
        st.session_state.candidate_logged_in = False

    # If already authenticated (HR) or logged in (Candidate), just return True
    if st.session_state.authenticated or st.session_state.candidate_logged_in:
        return True

    # Initialize active_login_tab_selection if not present
    if "active_login_tab_selection" not in st.session_state:
        if not os.path.exists(USER_DB_FILE) or len(load_users()) == 0:
            st.session_state.active_login_tab_selection = "Register"
        else:
            st.session_state.active_login_tab_selection = "Login"

    st.title("🔐 ScreenerPro Access")
    st.markdown("Please select your access mode.")

    mode = st.radio(
        "Select Mode:",
        ("HR Mode", "Candidate Mode"),
        key="mode_selection",
        horizontal=True,
        index=0 if st.session_state.app_mode == "hr_mode" else 1 if st.session_state.app_mode == "candidate_mode" else 0
    )

    if mode == "HR Mode":
        st.session_state.app_mode = "hr_mode"
        tabs = st.tabs(["Login", "Register"])

        with tabs[0]: # Login tab
            st.subheader("🔐 HR Login")
            st.info("If you don't have an account, please go to the 'Register' option first.")
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", key="username_login")
                password = st.text_input("Password", type="password", key="password_login")
                submitted = st.form_submit_button("Login")

                if submitted:
                    users = load_users()
                    if username not in users:
                        st.error("❌ Invalid username or password. Please register if you don't have an account.")
                    else:
                        user_data = users[username]
                        if user_data["status"] == "disabled":
                            st.error("❌ Your account has been disabled. Please contact an administrator.")
                        elif check_password(password, user_data["password"]):
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.session_state.user_company = user_data.get("company", "N/A")
                            st.session_state.candidate_logged_in = False
                            st.success("✅ HR Login successful!")
                            load_session_data_from_firestore_rest(st.session_state.username) # Load data on HR login
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password.")
        
        with tabs[1]: # Register tab
            register_section()

    elif mode == "Candidate Mode":
        st.session_state.app_mode = "candidate_mode"
        st.subheader("Candidate Login")
        if not st.session_state.candidate_logged_in:
            candidate_password = st.text_input("Enter Candidate Access Code", type="password", key="candidate_access_code")
            if st.button("Login as Candidate"):
                if candidate_password == "candidate123": # Placeholder: Replace with a more secure method
                    st.session_state.candidate_logged_in = True
                    st.session_state.authenticated = False
                    st.session_state.username = "Candidate User"
                    st.session_state.user_company = "N/A"
                    st.success("Logged in as Candidate!")
                    st.rerun()
                else:
                    st.error("Incorrect Access Code. Please try again.")
        else:
            st.info("You are already logged in as a Candidate.")
            if st.button("Logout from Candidate Mode"):
                st.session_state.candidate_logged_in = False
                st.session_state.app_mode = "select_mode"
                st.session_state.username = None
                st.session_state.user_company = None
                st.rerun()

    return st.session_state.authenticated or st.session_state.candidate_logged_in

def is_current_user_admin():
    return st.session_state.get("authenticated", False) and st.session_state.get("username") in ADMIN_USERNAME

def log_activity_main(message):
    if 'activity_log' not in st.session_state:
        st.session_state.activity_log = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.activity_log.insert(0, f"[{timestamp}] {message}")
    st.session_state.activity_log = st.session_state.activity_log[:50]

# --- Page Config ---
st.set_page_config(page_title="ScreenerPro – AI Hiring Dashboard", layout="wide", page_icon="�")

def load_css(css_file_name):
    try:
        with open(css_file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{css_file_name}' not found. Please ensure it's in the same directory as main.py.")

# Load the external CSS file
load_css("style.css")

# --- Dark Mode Toggle ---
dark_mode = st.sidebar.toggle("🌙 Dark Mode", key="dark_mode_main")

# --- Global Fonts & UI Styling ---
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
/* Hide GitHub fork button, Streamlit menu and footer */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {'#1E1E1E' if dark_mode else '#F0F2F6'};
    color: {'#E0E0E0' if dark_mode else '#333333'};
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
    color: {'#00cec9' if dark_mode else '#00cec9'};
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
@keyframes slideInDownFadeIn {{
    0% {{ opacity: 0; transform: translateY(-20px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}
.greeting-message {{
    font-size: 1.5rem;
    font-weight: 600;
    color: {'#00cec9' if dark_mode else '#00cec9'};
    margin-bottom: 1.5rem;
    animation: slideInDownFadeIn 0.7s ease-out;
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
    box_shadow: 0 4px 12px rgba(0,0,0,0.1);
}}
.stButton>button:hover {{
    background-color: #00b0a8;
    transform: translateY(-2px);
    box_shadow: 0 6px 16px rgba(0,0,0,0.15);
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
    sns.set_palette("viridis")
else:
    plt.style.use('default')
    sns.set_palette("coolwarm")

# ======================
# Analytics Dashboard Page Function (kept in main.py)
# ======================
def analytics_dashboard_page():
    st.markdown("""
    <style>
    .analytics-box {
        padding: 2rem;
        background: rgba(255, 255, 255, 0.96);
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        animation: fadeInSlide 0.7s ease-in-out;
        margin-bottom: 2rem;
    }
    @keyframes fadeInSlide {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    h3 {
        color: #00cec9;
        font-weight: 700;
    }
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="analytics-box">', unsafe_allow_html=True)
    st.markdown("## 📊 Screening Analytics Dashboard")

    def load_screening_data():
        """Loads screening results only from session state."""
        if 'comprehensive_df' in st.session_state and not st.session_state['comprehensive_df'].empty:
            try:
                st.info("✅ Loaded screening results from current session.")
                return st.session_state['comprehensive_df'].copy()
            except Exception as e:
                st.error(f"Error loading results from session state: {e}")
                return pd.DataFrame()
        else:
            st.warning("⚠️ No screening data found in current session. Please run the screener first.")
            return pd.DataFrame()

    df = load_screening_data()

    if df.empty:
        st.info("No data available for analytics. Please screen some resumes first.")
        st.stop()

    essential_core_columns = ['Score (%)', 'Years Experience', 'File Name', 'Candidate Name']
    missing_essential_columns = [col for col in essential_core_columns if col not in df.columns]

    if missing_essential_columns:
        st.error(f"Error: The loaded data is missing essential core columns: {', '.join(missing_essential_columns)}."
                 " Please ensure your screening process generates at least these required data fields.")
        st.stop()

    df['CGPA (4.0 Scale)'] = pd.to_numeric(df['CGPA (4.0 Scale)'], errors='coerce')
    df['CGPA (4.0 Scale)'] = df['CGPA (4.0 Scale)'].astype(float)

    df['Years Experience'] = pd.to_numeric(df['Years Experience'], errors='coerce')
    df['Score (%)'] = pd.to_numeric(df['Score (%)'], errors='coerce')
    
    df.replace('Not Found', pd.NA, inplace=True)
    df.replace('', pd.NA, inplace=True)

    st.markdown("### 🔍 Filter Analytics Data")
    col_filter1, col_filter2 = st.columns(2)

    with col_filter1:
        min_score, max_score = float(df['Score (%)'].min()), float(df['Score (%)'].max())
        score_range = st.slider(
            "Filter by Score (%)",
            min_value=min_score,
            max_value=max_score,
            value=(min_score, max_score),
            step=1.0,
            key="ana_score_filter"
        )

    with col_filter2:
        min_exp, max_exp = float(df['Years Experience'].min()), float(df['Years Experience'].max())
        exp_range = st.slider(
            "Filter by Years Experience",
            min_value=min_exp,
            max_value=max_exp,
            value=(min_exp, max_exp),
            step=0.5,
            key="ana_exp_filter"
        )
    
    col_filter3, col_filter4 = st.columns(2)
    with col_filter3:
        all_locations = sorted(list(df['Location'].dropna().unique()))
        selected_locations_ana = st.multiselect(
            "Filter by Location:",
            options=all_locations,
            key="ana_location_filter"
        )
    with col_filter4:
        all_languages_from_df = sorted(list(set(
            lang.strip() for langs_str in df['Languages'].dropna() if langs_str != "Not Found" for lang in langs_str.split(',')
        )))
        selected_languages_ana = st.multiselect(
            "Filter by Language:",
            options=all_languages_from_df,
            key="ana_language_filter"
        )

    filtered_df = df[
        (df['Score (%)'] >= score_range[0]) & (df['Score (%)'] <= score_range[1]) &
        (df['Years Experience'] >= exp_range[0]) & (df['Years Experience'] <= exp_range[1])
    ].copy()

    if selected_locations_ana:
        location_pattern = '|'.join([re.escape(loc) for loc in selected_locations_ana])
        filtered_df = filtered_df[
            filtered_df['Location'].fillna('').str.contains(location_pattern, case=False, na=False)
        ]
    
    if selected_languages_ana:
        language_pattern = '|'.join([re.escape(lang) for lang in selected_languages_ana])
        filtered_df = filtered_df[
            filtered_df['Languages'].fillna('').str.contains(language_pattern, case=False, na=False)
        ]

    if filtered_df.empty:
        st.warning("No data matches the current filter criteria. Adjust filters or upload more resumes.")
        st.stop()

    shortlist_threshold = st.session_state.get("screening_cutoff_score", 80)

    filtered_df['Shortlisted'] = filtered_df['Score (%)'].apply(lambda x: f"Yes (Score >= {shortlist_threshold}%)" if x >= shortlist_threshold else "No")

    st.markdown("### 📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg. Score", f"{filtered_df['Score (%)'].mean():.2f}%")
    col2.metric("Avg. Experience", f"{filtered_df['Years Experience'].mean():.1f} yrs")
    col3.metric("Total Candidates", f"{len(filtered_df)}")
    shortlisted_count_filtered = (filtered_df['Score (%)'] >= shortlist_threshold).sum()
    col4.metric("Shortlisted", f"{shortlisted_count_filtered}")

    st.divider()

    st.markdown("### 📋 Filtered Candidates List")
    display_cols_for_table = ['File Name', 'Candidate Name', 'Score (%)', 'Years Experience', 'Shortlisted']

    if 'Matched Keywords' in filtered_df.columns: display_cols_for_table.append('Matched Keywords')
    if 'Missing Skills' in filtered_df.columns: display_cols_for_table.append('Missing Skills')
    if 'AI Suggestion' in filtered_df.columns: display_cols_for_table.append('AI Suggestion')
    if 'Email' in filtered_df.columns: display_cols_for_table.append('Email')
    if 'Phone Number' in filtered_df.columns: display_cols_for_table.append('Phone Number')
    if 'Location' in filtered_df.columns: display_cols_for_table.append('Location')
    if 'Languages' in filtered_df.columns: display_cols_for_table.append('Languages')
    if 'Education Details' in filtered_df.columns: display_cols_for_table.append('Education Details')
    if 'Work History' in filtered_df.columns: display_cols_for_table.append('Work History')
    if 'Project Details' in filtered_df.columns: display_cols_for_table.append('Project Details')
    if 'CGPA (4.0 Scale)' in filtered_df.columns: display_cols_for_table.append('CGPA (4.0 Scale)')
    if 'Semantic Similarity' in filtered_df.columns: display_cols_for_table.append('Semantic Similarity')
    if 'Tag' in filtered_df.columns: display_cols_for_table.append('Tag')
    if 'JD Used' in filtered_df.columns: display_cols_for_table.append('JD Used')

    st.dataframe(
        filtered_df[display_cols_for_table].sort_values(by="Score (%)", ascending=False),
        use_container_width=True
    )

    @st.cache_data
    def convert_df_to_csv(df_to_convert):
        return df_to_convert.to_csv(index=False).encode('utf-8')

    csv = convert_df_to_csv(filtered_df)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_screening_results.csv",
        mime="text/csv",
        help="Download the data currently displayed in the table above."
    )

    st.divider()

    st.markdown("### 📊 Visualizations")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "Score Distribution", "Experience Distribution", "Shortlist Breakdown",
        "Score vs. Experience", "Skill Clouds", "CGPA Distribution",
        "Score vs. CGPA", "Experience vs. CGPA", "Skills by Category",
        "Location Distribution"
    ])

    with tab1:
        st.markdown("#### Score Distribution")
        fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
        sns.histplot(filtered_df['Score (%)'], bins=10, kde=True, color="#00cec9", ax=ax_hist)
        ax_hist.set_xlabel("Score (%)")
        ax_hist.set_ylabel("Number of Candidates")
        st.pyplot(fig_hist)
        plt.close(fig_hist)

    with tab2:
        st.markdown("#### Experience Distribution")
        fig_exp, ax_exp = plt.subplots(figsize=(10, 5))
        sns.histplot(filtered_df['Years Experience'], bins=5, kde=True, color="#fab1a0", ax=ax_exp)
        ax_exp.set_xlabel("Years of Experience")
        ax_exp.set_ylabel("Number of Candidates")
        st.pyplot(fig_exp)
        plt.close(fig_exp)

    with tab3:
        st.markdown("#### Shortlist Breakdown")
        shortlist_counts = filtered_df['Shortlisted'].value_counts()
        if not shortlist_counts.empty:
            fig_pie = px.pie(
                names=shortlist_counts.index,
                values=shortlist_counts.values,
                title=f"Candidates Shortlisted vs. Not Shortlisted (Cutoff: {shortlist_threshold}%)",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Not enough data to generate Shortlist Breakdown.")

    with tab4:
        st.markdown("#### Score vs. Years Experience")
        fig_scatter = px.scatter(
            filtered_df,
            x="Years Experience",
            y="Score (%)",
            hover_name="Candidate Name",
            color="Shortlisted",
            title="Candidate Score vs. Years Experience",
            labels={"Years Experience": "Years of Experience", "Score (%)": "Matching Score (%)"},
            trendline="ols",
            color_discrete_map={f"Yes (Score >= {shortlist_threshold}%)": "green", "No": "red"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab5:
        col_wc1, col_wc2 = st.columns(2)
        with col_wc1:
            st.markdown("#### ☁️ Common Skills WordCloud")
            if 'Matched Keywords' in filtered_df.columns and not filtered_df['Matched Keywords'].empty:
                all_keywords = [
                    kw.strip() for kws in filtered_df['Matched Keywords'].dropna()
                    for kw in str(kws).split(',') if kw.strip()
                ]
                if all_keywords:
                    wc = WordCloud(width=800, height=400, background_color="white", collocations=False).generate(" ".join(all_keywords))
                    fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
                    ax_wc.imshow(wc, interpolation='bilinear')
                    ax_wc.axis('off')
                    st.pyplot(fig_wc)
                    plt.close(fig_wc)
                else:
                    st.info("No common skills to display in the WordCloud for filtered data.")
            else:
                st.info("No 'Matched Keywords' data available or column not found for WordCloud.")

        with col_wc2:
            st.markdown("#### ❌ Top Missing Skills")
            if 'Missing Skills' in filtered_df.columns and not filtered_df['Missing Skills'].empty:
                all_missing = pd.Series([
                    s.strip() for row in filtered_df['Missing Skills'].dropna()
                    for s in str(row).split(',') if s.strip()
                ])
                if not all_missing.empty:
                    sns.set_style("whitegrid")
                    fig_ms, ax_ms = plt.subplots(figsize=(8, 4))
                    top_missing = all_missing.value_counts().head(10)
                    sns.barplot(x=top_missing.values, y=top_missing.index, ax=ax_ms, palette="coolwarm")
                    ax_ms.set_xlabel("Count")
                    ax_ms.set_ylabel("Missing Skill")
                    st.pyplot(fig_ms)
                    plt.close(fig_ms)
                else:
                    st.info("No top missing skills to display for filtered data.")
            else:
                st.info("No 'Missing Skills' data available or column not found.")

    with tab6:
        st.markdown("#### 🎓 CGPA Distribution")
        cgpa_df = filtered_df.dropna(subset=['CGPA (4.0 Scale)'])
        if not cgpa_df.empty:
            fig_cgpa_hist = px.histogram(
                cgpa_df,
                x='CGPA (4.0 Scale)',
                nbins=10,
                title='Distribution of CGPA (Normalized to 4.0 Scale)',
                labels={'CGPA (4.0 Scale)': 'CGPA'},
                color_discrete_sequence=px.colors.qualitative.Plotly if not dark_mode else px.colors.qualitative.Dark2
            )
            st.plotly_chart(fig_cgpa_hist, use_container_width=True)
        else:
            st.info("No CGPA data available for this visualization after filtering.")

    with tab7:
        st.markdown("#### 📈 Score vs. CGPA")
        cgpa_df_scatter = filtered_df.dropna(subset=['CGPA (4.0 Scale)'])
        if not cgpa_df_scatter.empty:
            fig_score_cgpa = px.scatter(
                cgpa_df_scatter,
                x='CGPA (4.0 Scale)',
                y='Score (%)',
                hover_name='Candidate Name',
                color='Shortlisted',
                title='Candidate Score vs. CGPA',
                labels={'CGPA (4.0 Scale)': 'CGPA (4.0 Scale)', 'Score (%)': 'Matching Score (%)'},
                trendline="ols",
                color_discrete_map={f"Yes (Score >= {shortlist_threshold}%)": "green", "No": "red"}
            )
            st.plotly_chart(fig_score_cgpa, use_container_width=True)
        else:
            st.info("No CGPA data available for this visualization after filtering.")

    with tab8:
        st.markdown("#### 📊 Experience vs. CGPA")
        cgpa_df_exp_scatter = filtered_df.dropna(subset=['CGPA (4.0 Scale)'])
        if not cgpa_df_exp_scatter.empty:
            fig_exp_cgpa = px.scatter(
                cgpa_df_exp_scatter,
                x='Years Experience',
                y='CGPA (4.0 Scale)',
                hover_name='Candidate Name',
                color='Shortlisted',
                title='Years Experience vs. CGPA',
                labels={'Years Experience': 'Years of Experience', 'CGPA (4.0 Scale)': 'CGPA (4.0 Scale)'},
                trendline="ols",
                color_discrete_map={f"Yes (Score >= {shortlist_threshold}%)": "green", "No": "red"}
            )
            st.plotly_chart(fig_exp_cgpa, use_container_width=True)
        else:
            st.info("No CGPA data available for this visualization after filtering.")

    with tab9:
        st.markdown("#### 🧠 Skills by Category")
        if 'Matched Keywords (Categorized)' in filtered_df.columns and not filtered_df['Matched Keywords (Categorized)'].empty:
            all_categorized_skills_counts = collections.defaultdict(int)
            for categorized_dict in filtered_df['Matched Keywords (Categorized)'].dropna():
                if isinstance(categorized_dict, dict):
                    for category, skills_list in categorized_dict.items():
                        all_categorized_skills_counts[category] += len(skills_list)

            if all_categorized_skills_counts:
                skills_cat_df = pd.DataFrame(all_categorized_skills_counts.items(), columns=['Category', 'Count']).sort_values('Count', ascending=False)
                fig_skills_cat = px.bar(
                    skills_cat_df,
                    x='Count',
                    y='Category',
                    orientation='h',
                    title='Total Matched Skills by Category',
                    labels={'Count': 'Number of Matched Skills', 'Category': 'Skill Category'},
                    color='Count',
                    color_continuous_scale=px.colors.sequential.Teal if not dark_mode else px.colors.sequential.Plasma
                )
                st.plotly_chart(fig_skills_cat, use_container_width=True)
            else:
                st.info("No categorized skill data available for this visualization.")
        else:
            st.info("No 'Matched Keywords (Categorized)' data available or column not found.")

    with tab10:
        st.markdown("#### 📍 Candidate Location Distribution")
        if 'Location' in filtered_df.columns and not filtered_df['Location'].empty:
            all_locations = []
            for loc_str in filtered_df['Location'].dropna():
                all_locations.extend([loc.strip() for loc in loc_str.split(',') if loc.strip() and loc.strip().lower() != 'not found'])

            if all_locations:
                location_counts = pd.Series(all_locations).value_counts().reset_index()
                location_counts.columns = ['Location', 'Count']
                fig_location = px.bar(
                    location_counts,
                    x='Count',
                    y='Location',
                    orientation='h',
                    title='Candidate Distribution by Location',
                    labels={'Count': 'Number of Candidates', 'Location': 'Location'},
                    color='Count',
                    color_continuous_scale=px.colors.sequential.Viridis if not dark_mode else px.colors.sequential.Cividis
                )
                st.plotly_chart(fig_location, use_container_width=True)
            else:
                st.info("No valid location data available for this visualization.")
        else:
            st.info("No 'Location' data available or column not found.")


    st.markdown("</div>", unsafe_allow_html=True)


# Main application logic
def main_app():
    # Initialize session states for app mode and login status if not present
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = "select_mode"
    if 'candidate_logged_in' not in st.session_state:
        st.session_state.candidate_logged_in = False
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    # If the mode is 'select_mode', show the login/mode selection UI
    if st.session_state.app_mode == "select_mode":
        login_section() # This function handles mode selection and sets session states
    
    # After login_section (which might rerun the app), check the app_mode again
    if st.session_state.app_mode == "hr_mode":
        if not st.session_state.authenticated: # Should ideally not happen due to rerun in login_section
            st.session_state.app_mode = "select_mode"
            st.rerun()
        
        # Log successful login (only once per session)
        if st.session_state.get('last_login_logged_for_user') != st.session_state.username:
            log_activity_main(f"User '{st.session_state.username}' logged in to HR Mode.")
            st.session_state.last_login_logged_for_user = st.session_state.username

        is_admin = is_current_user_admin()

        # Initialize comprehensive_df globally if it doesn't exist
        if 'comprehensive_df' not in st.session_state:
            st.session_state['comprehensive_df'] = pd.DataFrame()

        st.markdown(f'<div class="greeting-message">Hello, {st.session_state.username}! 👋</div>', unsafe_allow_html=True)

        navigation_options_hr = [
            "🏠 Dashboard", "🧠 Resume Screener", "📁 Manage JDs", "📊 Screening Analytics",
            "📦 Bulk Resume Import",
            "📤 Email Candidates", "🔍 Search Resumes", "📝 Candidate Notes",
            "📈 Advanced Tools",
            "🤝 Collaboration Hub",
            "🏢 About Us",
            "⚖️ Privacy Policy & Terms",
            "❓ Feedback & Help"
        ]
        if is_admin:
            navigation_options_hr.append("⚙️ Admin Tools")
        navigation_options_hr.append("🚪 Logout")

        default_tab_hr = st.session_state.get("tab_override", "🏠 Dashboard")
        if default_tab_hr not in navigation_options_hr:
            default_tab_hr = "🏠 Dashboard"
        tab_hr = st.sidebar.radio("📍 Navigate (HR Mode)", navigation_options_hr, index=navigation_options_hr.index(default_tab_hr))
        if "tab_override" in st.session_state:
            del st.session_state.tab_override

        # HR Page Routing
        if tab_hr == "🏠 Dashboard":
            st.markdown('<div class="dashboard-header">📊 Overview Dashboard</div>', unsafe_allow_html=True)

            resume_count = 0
            if not os.path.exists("data"):
                os.makedirs("data")
            jd_count = len([f for f in os.listdir("data") if f.endswith(".txt")])
            shortlisted = 0
            avg_score = 0.0
            df_results = pd.DataFrame()

            cutoff_score = st.session_state.get('screening_cutoff_score', 75)
            min_exp_required = st.session_state.get('screening_min_experience', 2)
            max_exp_allowed = st.session_state.get('screening_max_experience', 10)
            min_cgpa_required = st.session_state.get('screening_min_cgpa', 2.5)

            if 'comprehensive_df' in st.session_state and not st.session_state['comprehensive_df'].empty:
                try:
                    df_results = st.session_state['comprehensive_df'].copy()
                    resume_count = df_results["File Name"].nunique()

                    df_results['CGPA (4.0 Scale)'] = pd.to_numeric(df_results['CGPA (4.0 Scale)'], errors='coerce')
                    df_results['CGPA (4.0 Scale)'] = df_results['CGPA (4.0 Scale)'].astype(float)

                    shortlisted_df = df_results[
                        (df_results["Score (%)"] >= cutoff_score) &
                        (df_results["Years Experience"] >= min_exp_required) &
                        (df_results["Years Experience"] <= max_exp_allowed) &
                        ((df_results['CGPA (4.0 Scale)'].isnull()) | (df_results['CGPA (4.0 Scale)'] >= min_cgpa_required))
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

            st.subheader("Key Performance Indicators")
            metric_cols = st.columns(4)

            metric_cols[0].metric("Resumes Screened", resume_count, help="Total unique resumes processed in this session.")
            metric_cols[1].metric("Job Descriptions", jd_count, help="Number of job descriptions available.")
            metric_cols[2].metric("Shortlisted Candidates", shortlisted, help=f"Candidates meeting Score ≥ {cutoff_score}%, Exp {min_exp_required}-{max_exp_allowed} yrs, CGPA ≥ {min_cgpa_required} or N/A.")
            metric_cols[3].metric("Average Score", f"{avg_score:.1f}%", help="Average matching score of all screened resumes.")

            st.markdown("---")

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
            st.subheader("Cloud Session Data Management")
            cloud_data_cols = st.columns(2)
            with cloud_data_cols[0]:
                if st.button("💾 Save Session Data to Cloud (REST API)", key="save_session_data_button"):
                    save_session_data_to_firestore_rest(st.session_state.get('username', 'anonymous'))
            with cloud_data_cols[1]:
                if st.button("🔄 Load Session Data from Cloud (REST API)", key="load_session_data_button"):
                    load_session_data_from_firestore_rest(st.session_state.get('username', 'anonymous'))
                    st.rerun()

            st.markdown("---")

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

            if not df_results.empty:
                try:
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
                            fig_plotly_pie = px.pie(pie_data, values='Count', names='Tag', title='Candidate Quality Breakdown',
                                                    color_discrete_sequence=px.colors.qualitative.Pastel if not dark_mode else px.colors.qualitative.Dark2)
                            st.plotly_chart(fig_plotly_pie, use_container_width=True)

                    if st.session_state.dashboard_widgets['Experience Distribution']:
                        with col_g2:
                            st.markdown("##### 📊 Experience Level Breakdown")
                            bins = [0, 2, 5, 10, 20, 50]
                            labels = ['0-2 yrs', '3-5 yrs', '6-10 yrs', '10-20 yrs', '20+ yrs']
                            df_results['Experience Group'] = pd.cut(df_results['Years Experience'], bins=bins, labels=labels, right=False)
                            exp_counts = df_results['Experience Group'].value_counts().sort_index()

                            fig_plotly_bar = px.bar(exp_counts, x=exp_counts.index, y=exp_counts.values, title='Experience Distribution',
                                                    labels={'x': 'Experience Range', 'y': 'Number of Candidates'},
                                                    color_discrete_sequence=px.colors.sequential.Plasma if dark_mode else px.colors.sequential.Viridis)
                            st.plotly_chart(fig_plotly_bar, use_container_width=True)

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

            if st.session_state.dashboard_widgets['My Recent Screenings']:
                st.subheader("My Recent Screenings")
                if not df_results.empty:
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

                    if 'JD Used' not in df_all_results.columns:
                        df_all_results['JD Used'] = 'Default Job Description'
                        st.warning("Note: 'JD Used' column not found in screening results. Using 'Default Job Description' for display. Please update your screener to track the JD used.")

                    if 'JD Used' in df_all_results.columns:
                        df_all_results['CGPA (4.0 Scale)'] = pd.to_numeric(df_all_results['CGPA (4.0 Scale)'], errors='coerce')
                        df_all_results['CGPA (4.0 Scale)'] = df_all_results['CGPA (4.0 Scale)'].astype(float)

                        shortlisted_per_jd = df_all_results[
                            (df_all_results["Score (%)"] >= cutoff_score) &
                            (df_all_results["Years Experience"] >= min_exp_required) &
                            (df_all_results["Years Experience"] <= max_exp_allowed) &
                            ((df_all_results['CGPA (4.0 Scale)'].isnull()) | (df_all_results['CGPA (4.0 Scale)'] >= min_cgpa_required))
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
                        log_activity_main(f"Mock candidate '{mock_candidate['candidate_name']}' added for approval.")
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
                                        log_activity_main(f"Candidate '{candidate['candidate_name']}' approved.")
                                        st.success(f"Approved {candidate['candidate_name']}!")
                                        st.rerun()
                                with col_reject:
                                    if st.button(f"❌ Reject {candidate['candidate_name']}", key=f"reject_{i}"):
                                        st.session_state.pending_approvals[i]['status'] = 'rejected'
                                        log_activity_main(f"Candidate '{candidate['candidate_name']}' rejected.")
                                        st.error(f"Rejected {candidate['candidate_name']}.")
                                        st.rerun()
                    approved_rejected = [c for c in st.session_state.pending_approvals if c['status'] != 'pending']
                    if approved_rejected:
                        st.markdown("---")
                        st.subheader("Reviewed Candidates")
                        reviewed_df = pd.DataFrame(approved_rejected)
                        st.dataframe(reviewed_df[['candidate_name', 'score', 'experience', 'status']], use_container_width=True, hide_index=True)

        elif tab_hr == "🧠 Resume Screener":
            resume_screener_page()
            if 'comprehensive_df' in st.session_state and not st.session_state['comprehensive_df'].empty:
                current_df_len = len(st.session_state['comprehensive_df'])
                if st.session_state.get('last_screen_log_count', 0) < current_df_len:
                    log_activity_main(f"Performed resume screening for {current_df_len} candidates.")
                    st.session_state.last_screen_log_count = current_df_len

                for result in st.session_state['comprehensive_df'].to_dict('records'):
                    if result.get('Score (%)', 0) >= 90 and result['Candidate Name'] not in [app['candidate_name'] for app in st.session_state.get('pending_approvals', []) if app['status'] == 'pending']:
                        if 'pending_approvals' not in st.session_state:
                            st.session_state.pending_approvals = []
                        st.session_state.pending_approvals.append({
                            "candidate_name": result['Candidate Name'],
                            "score": result['Score (%)'],
                            "experience": result['Years Experience'],
                            "jd_used": result.get('JD Used', 'N/A'),
                            "status": "pending",
                            "notes": f"High-scoring candidate from recent screening."
                        })
                        log_activity_main(f"Candidate '{result['Candidate Name']}' sent for approval (high score).")
                        st.toast(f"Candidate {result['Candidate Name']} sent for approval!")

        elif tab_hr == "📁 Manage JDs":
            try:
                with open("manage_jds.py", encoding="utf-8") as f:
                    exec(f.read())
            except FileNotFoundError:
                st.info("`manage_jds.py` not found. Please ensure the file exists in the same directory.")
            except Exception as e:
                st.error(f"Error loading Manage JDs: {e}")

        elif tab_hr == "📊 Screening Analytics":
            analytics_dashboard_page()

        elif tab_hr == "📈 Advanced Tools":
            from advanced import advanced_tools_page
            advanced_tools_page(
                app_id=FIREBASE_PROJECT_ID,
                FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY,
                FIRESTORE_BASE_URL=FIRESTORE_BASE_URL
            )
        elif tab_hr == "🤝 Collaboration Hub":
            from collaboration import collaboration_hub_page
            collaboration_hub_page(
                app_id=FIREBASE_PROJECT_ID,
                FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY,
                FIRESTORE_BASE_URL=FIRESTORE_BASE_URL
            )

        elif tab_hr == "🏢 About Us":
            about_us_page()

        elif tab_hr == "⚖️ Privacy Policy & Terms":
            privacy_policy_page()

        elif tab_hr == "📤 Email Candidates":
            try:
                from email_sender import send_email_to_candidate
                send_email_to_candidate()
            except ImportError:
                st.error("`email_sender.py` not found or `send_email_to_candidate` function not defined. Please ensure 'email_sender.py' exists and contains the 'send_email_to_candidate' function.")
            except Exception as e:
                st.error(f"Error loading Email Candidates: {e}")

        elif tab_hr == "🔍 Search Resumes":
            try:
                with open("search.py", encoding="utf-8") as f:
                    exec(f.read())
            except FileNotFoundError:
                st.info("`search.py` not found. Please ensure the file exists in the same directory.")
            except Exception as e:
                st.error(f"Error loading Search Resumes: {e}")

        elif tab_hr == "📝 Candidate Notes":
            try:
                with open("notes.py", encoding="utf-8") as f:
                    exec(f.read())
            except FileNotFoundError:
                st.info("`notes.py` not found. Please ensure the file exists in the same directory.")
            except Exception as e:
                st.error(f"Error loading Candidate Notes: {e}")

        elif tab_hr == "❓ Feedback & Help":
            from feedback import feedback_and_help_page
            if 'user_email' not in st.session_state:
                st.session_state['user_email'] = st.session_state.get('username', 'anonymous_user')
            feedback_and_help_page()
        elif tab_hr == "📦 Bulk Resume Import":
            from bulk_upload_page import bulk_upload_page
            jd_texts = {}
            if os.path.exists("data"):
                for fname in os.listdir("data"):
                    if fname.endswith(".txt"):
                        with open(os.path.join("data", fname), "r", encoding="utf-8") as f:
                            jd_texts[fname.replace(".txt", "").replace("_", " ").title()] = f.read()
            bulk_upload_page(st.session_state['comprehensive_df'], jd_texts)

        elif tab_hr == "⚙️ Admin Tools":
            st.markdown('<div class="dashboard-header">⚙️ Admin Tools</div>', unsafe_allow_html=True)
            if is_admin:
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

        elif tab_hr == "🚪 Logout":
            log_activity_main(f"User '{st.session_state.get('username', 'anonymous_user')}' logged out from HR Mode.")
            st.session_state.authenticated = False
            st.session_state.pop('username', None)
            st.session_state.pop('user_company', None)
            st.session_state.app_mode = "select_mode"
            st.success("✅ Logged out.")
            st.rerun()

    elif st.session_state.app_mode == "candidate_mode":
        if not st.session_state.candidate_logged_in:
            st.session_state.app_mode = "select_mode"
            st.rerun()
        
        if st.session_state.get('last_login_logged_for_candidate') != st.session_state.username:
            log_activity_main(f"Candidate '{st.session_state.username}' logged in to Candidate Mode.")
            st.session_state.last_login_logged_for_candidate = st.session_state.username

        st.markdown(f'<div class="greeting-message">Hello, Candidate! 👋</div>', unsafe_allow_html=True)

        navigation_options_candidate = [
            "🧠 My Resume Screener",
            "🏢 About Us",
            "⚖️ Privacy Policy & Terms",
            "❓ Feedback & Help",
            "🚪 Logout"
        ]
        tab_candidate = st.sidebar.radio("📍 Navigate (Candidate Mode)", navigation_options_candidate)

        if tab_candidate == "🧠 My Resume Screener":
            candidate_screener_page() # Call the candidate-specific screener page
        elif tab_candidate == "🏢 About Us":
            about_us_page()
        elif tab_candidate == "⚖️ Privacy Policy & Terms":
            privacy_policy_page()
        elif tab_candidate == "❓ Feedback & Help":
            from feedback import feedback_and_help_page
            if 'user_email' not in st.session_state:
                st.session_state['user_email'] = st.session_state.get('username', 'anonymous_user')
            feedback_and_help_page()
        elif tab_candidate == "🚪 Logout":
            log_activity_main(f"Candidate '{st.session_state.get('username', 'anonymous_user')}' logged out from Candidate Mode.")
            st.session_state.candidate_logged_in = False
            st.session_state.app_mode = "select_mode"
            st.session_state.pop('username', None)
            st.session_state.pop('user_company', None)
            st.success("✅ Logged out.")
            st.rerun()

# Call the main application function
if __name__ == "__main__":
    main_app()
�
