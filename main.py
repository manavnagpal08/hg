import streamlit as st
import json
import bcrypt
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from datetime import datetime
import plotly.express as px
import statsmodels.api as sm
import collections

# Firebase imports
import firebase_admin
from firebase_admin import credentials, initialize_app, firestore, get_app

# --- Firebase Initialization (Safe Check) ---
import firebase_admin
from firebase_admin import credentials, initialize_app, firestore

try:
    # Set environment variable for Application Default Credentials
    key_path = os.path.abspath("config/firebase-key.json")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

    if not firebase_admin._apps:
        cred = credentials.ApplicationDefault()
        initialize_app(cred)

    db = firestore.client()
    st.session_state.db = db
except Exception as e:
    st.warning(f"🔥 Firebase init failed: {e}")
    st.session_state.db = None

# All remaining original code follows from here

# File to store user credentials
USER_DB_FILE = "users.json"
ADMIN_USERNAME = ("admin@forscreenerpro", "admin@forscreenerpro2")

# [...rest of your main.py code remains unchanged...]

# If you want me to re-paste the entire long file inline with this fixed header, just say "yes full paste" and I’ll drop the whole thing for you.
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
            elif "status" not in data:
                data["status"] = "active"
            if "company" not in data: # Add company field if missing
                data["company"] = "N/A"
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
        new_company_name = st.text_input("Company Name", key="new_company_name_reg_public") # New field
        new_password = st.text_input("Choose Password", type="password", key="new_password_reg_public")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password_reg_public")
        register_button = st.form_submit_button("Register New Account")

        if register_button:
            if not new_username or not new_password or not confirm_password or not new_company_name:
                st.error("Please fill in all fields.")
            elif not is_valid_email(new_username): # Email format validation
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
                        "company": new_company_name # Store company name
                    }
                    save_users(users)
                    st.success("✅ Registration successful! You can now switch to the 'Login' option.")
                    # Manually set the session state to switch to Login option
                    st.session_state.active_login_tab_selection = "Login"

def admin_registration_section():
    """Admin-driven user creation form."""
    st.subheader("➕ Create New User Account (Admin Only)")
    with st.form("admin_registration_form", clear_on_submit=True):
        new_username = st.text_input("New User's Username (Email)", key="new_username_admin_reg")
        new_company_name = st.text_input("New User's Company Name", key="new_company_name_admin_reg") # New field
        new_password = st.text_input("New User's Password", type="password", key="new_password_admin_reg")
        admin_register_button = st.form_submit_button("Add New User")

    if admin_register_button:
        if not new_username or not new_password or not new_company_name:
            st.error("Please fill in all fields.")
        elif not is_valid_email(new_username): # Email format validation
            st.error("Please enter a valid email address for the username.")
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
    # Exclude all admin usernames from the list of users whose passwords can be reset
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
    # Exclude all admin usernames from the list of users whose status can be toggled
    user_options = [user for user in users.keys() if user not in ADMIN_USERNAME]

    if not user_options:
        st.info("No other users to manage status for.")
        return

    with st.form("admin_toggle_user_status_form", clear_on_submit=False): # Keep values after submit for easier toggling
        selected_user = st.selectbox("Select User to Toggle Status", user_options, key="toggle_user_select")

        current_status = users[selected_user]["status"]
        st.info(f"Current status of '{selected_user}': **{current_status.upper()}**")

        if st.form_submit_button(f"Toggle to {'Disable' if current_status == 'active' else 'Enable'} User"):
            new_status = "disabled" if current_status == "active" else "active"
            users[selected_user]["status"] = new_status
            save_users(users)
            st.success(f"✅ User '{selected_user}' status set to **{new_status.upper()}**.")
            st.rerun() # Rerun to update the displayed status immediately


# --- Firebase Data Persistence Functions ---
def save_session_data_to_firestore():
    """
    Saves key session state data (like comprehensive_df) to Firestore for the current user.
    """
    if st.session_state.get('db') and st.session_state.get('username'):
        db = st.session_state.db
        user_id = st.session_state.username # Using username as user_id for simplicity
        app_id = globals().get('__app_id', 'default-app-id')

        # Document reference for user-specific session data
        # Data stored in /artifacts/{appId}/users/{userId}/session_data/current_session
        doc_ref = db.collection(f'artifacts/{app_id}/users/{user_id}/session_data').document('current_session')

        data_to_save = {}
        # Save comprehensive_df if it exists and is not empty
        if 'comprehensive_df' in st.session_state and not st.session_state['comprehensive_df'].empty:
            # Convert DataFrame to JSON string to store in a single Firestore field
            # This is necessary because Firestore doesn't directly support complex nested objects
            # or large arrays of maps without specific structuring.
            # Also, filter out 'Resume Raw Text' as it can be very large.
            df_for_save = st.session_state['comprehensive_df'].drop(columns=['Resume Raw Text'], errors='ignore')
            data_to_save['comprehensive_df_json'] = df_for_save.to_json(orient='records')

        # You can add other session variables here if needed, ensure they are JSON serializable
        # Example: data_to_save['last_jd_used'] = st.session_state.get('last_jd_used')
        # Example: data_to_save['screening_cutoff_score'] = st.session_state.get('screening_cutoff_score')

        if data_to_save:
            try:
                # This is one of the "4-5 lines of code" for saving
                doc_ref.set(data_to_save)
                st.toast("Session data saved to Firestore!")
                log_activity(f"Session data saved for user '{user_id}'.")
            except Exception as e:
                st.error(f"Error saving session data to Firestore: {e}")
        else:
            st.info("No relevant session data to save.")

def load_session_data_from_firestore():
    """
    Loads key session state data from Firestore for the current user.
    """
    if st.session_state.get('db') and st.session_state.get('username'):
        db = st.session_state.db
        user_id = st.session_state.username
        app_id = globals().get('__app_id', 'default-app-id')

        doc_ref = db.collection(f'artifacts/{app_id}/users/{user_id}/session_data').document('current_session')

        try:
            # This is one of the "4-5 lines of code" for loading
            doc = doc_ref.get()
            if doc.exists:
                data = doc.to_dict()
                if 'comprehensive_df_json' in data:
                    # Load DataFrame from JSON string
                    st.session_state['comprehensive_df'] = pd.read_json(data['comprehensive_df_json'], orient='records')
                    st.toast("Session data loaded from Firestore!")
                    log_activity(f"Session data loaded for user '{user_id}'.")

                # Load other session variables if they were saved
                # Example: st.session_state['last_jd_used'] = data.get('last_jd_used')
                # Example: st.session_state['screening_cutoff_score'] = data.get('screening_cutoff_score')

            else:
                st.info("No previous session data found in Firestore.")
        except Exception as e:
            st.error(f"Error loading session data from Firestore: {e}")
    else:
        st.warning("Cannot load session data: Firestore not initialized or user not logged in.")


def login_section():
    """Handles user login and public registration."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None

    # Initialize active_login_tab_selection if not present
    if "active_login_tab_selection" not in st.session_state:
        # Default to 'Register' if no users, otherwise 'Login'
        if not os.path.exists(USER_DB_FILE) or len(load_users()) == 0:
            st.session_state.active_login_tab_selection = "Register"
        else:
            st.session_state.active_login_tab_selection = "Login"


    if st.session_state.authenticated:
        return True

    # Use st.radio to simulate tabs if st.tabs() default_index is not supported
    tab_selection = st.radio(
        "Select an option:",
        ("Login", "Register"),
        key="login_register_radio",
        index=0 if st.session_state.active_login_tab_selection == "Login" else 1
    )

    if tab_selection == "Login":
        st.subheader("🔐 HR Login")
        st.info("If you don't have an account, please go to the 'Register' option first.") # Added instructional message
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
                        st.session_state.user_company = user_data.get("company", "N/A") # Store company name
                        st.success("✅ Login successful!")
                        # --- Load session data after successful login ---
                        load_session_data_from_firestore() # Call to load data
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password.")

    elif tab_selection == "Register": # This will be the initially selected option for new users
        register_section()

    return st.session_state.authenticated

# Helper function to check if the current user is an admin
def is_current_user_admin():
    # Check if the current username is in the ADMIN_USERNAME tuple
    return st.session_state.get("authenticated", False) and st.session_state.get("username") in ADMIN_USERNAME

# --- Helper for Activity Logging ---
def log_activity(message):
    """Logs an activity with a timestamp to the session state."""
    if 'activity_log' not in st.session_state:
        st.session_state.activity_log = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
# Analytics Dashboard Page Function
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
        100% {{ opacity: 1; transform: translateY(0); }}
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

    st.markdown("### 🔍 Filter Results")
    filter_cols = st.columns(3)

    with filter_cols[0]:
        min_score, max_score = float(df['Score (%)'].min()), float(df['Score (%)'].max())
        score_range = st.slider(
            "Filter by Score (%)",
            min_value=min_score,
            max_value=max_score,
            value=(min_score, max_score),
            step=1.0,
            key="score_filter"
        )

    with filter_cols[1]:
        min_exp, max_exp = float(df['Years Experience'].min()), float(df['Years Experience'].max())
        exp_range = st.slider(
            "Filter by Years Experience",
            min_value=min_exp,
            max_value=max_exp,
            value=(min_exp, max_exp),
            step=0.5,
            key="exp_filter"
        )

    with filter_cols[2]:
        shortlist_threshold = st.slider(
            "Set Shortlisting Cutoff Score (%)",
            min_value=0,
            max_value=100,
            value=80,
            step=1,
            key="shortlist_filter"
        )

    filtered_df = df[
        (df['Score (%)'] >= score_range[0]) & (df['Score (%)'] <= score_range[1]) &
        (df['Years Experience'] >= exp_range[0]) & (df['Years Experience'] <= exp_range[1])
    ].copy()

    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust your criteria.")
        st.stop()

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

    if 'Matched Keywords' in filtered_df.columns:
        display_cols_for_table.append('Matched Keywords')
    if 'Missing Skills' in filtered_df.columns:
        display_cols_for_table.append('Missing Skills')
    if 'AI Suggestion' in filtered_df.columns:
        display_cols_for_table.append('AI Suggestion')
    if 'Email' in filtered_df.columns:
        display_cols_for_table.append('Email')
    if 'Phone Number' in filtered_df.columns:
        display_cols_for_table.append('Phone Number')
    if 'Location' in filtered_df.columns:
        display_cols_for_table.append('Location')
    if 'Languages' in filtered_df.columns:
        display_cols_for_table.append('Languages')
    if 'Education Details' in filtered_df.columns:
        display_cols_for_table.append('Education Details')
    if 'Work History' in filtered_df.columns:
        display_cols_for_table.append('Work History')
    if 'Project Details' in filtered_df.columns:
        display_cols_for_table.append('Project Details')
    if 'CGPA (4.0 Scale)' in filtered_df.columns:
        display_cols_for_table.append('CGPA (4.0 Scale)')
    if 'Semantic Similarity' in filtered_df.columns:
        display_cols_for_table.append('Semantic Similarity')
    if 'Tag' in filtered_df.columns:
        display_cols_for_table.append('Tag')
    if 'JD Used' in filtered_df.columns:
        display_cols_for_table.append('JD Used')

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
                    wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(all_keywords))
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
        if 'CGPA (4.0 Scale)' in filtered_df.columns and not filtered_df['CGPA (4.0 Scale)'].isnull().all():
            fig_cgpa_hist = px.histogram(
                filtered_df.dropna(subset=['CGPA (4.0 Scale)']),
                x='CGPA (4.0 Scale)',
                nbins=10,
                title='Distribution of CGPA (Normalized to 4.0 Scale)',
                labels={'CGPA (4.0 Scale)': 'CGPA'},
                color_discrete_sequence=px.colors.qualitative.Plotly[0] if not dark_mode else px.colors.qualitative.Dark2[0]
            )
            st.plotly_chart(fig_cgpa_hist, use_container_width=True)
        else:
            st.info("No CGPA data available for this visualization.")

    with tab7:
        st.markdown("#### 📈 Score vs. CGPA")
        if 'CGPA (4.0 Scale)' in filtered_df.columns and not filtered_df['CGPA (4.0 Scale)'].isnull().all():
            fig_score_cgpa = px.scatter(
                filtered_df.dropna(subset=['CGPA (4.0 Scale)']),
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
            st.info("No CGPA data available for this visualization.")

    with tab8:
        st.markdown("#### 📊 Experience vs. CGPA")
        if 'CGPA (4.0 Scale)' in filtered_df.columns and not filtered_df['CGPA (4.0 Scale)'].isnull().all():
            fig_exp_cgpa = px.scatter(
                filtered_df.dropna(subset=['CGPA (4.0 Scale)']),
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
            st.info("No CGPA data available for this visualization.")

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


# ======================
# 🏠 Dashboard Section
# ======================
if tab == "🏠 Dashboard":
    st.markdown('<div class="dashboard-header">📊 Overview Dashboard</div>', unsafe_allow_html=True)

    # Initialize metrics
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

    # --- Add Save/Load Buttons for Session Data ---
    st.markdown("---")
    st.subheader("Cloud Session Data Management")
    cloud_data_cols = st.columns(2)
    with cloud_data_cols[0]:
        if st.button("💾 Save Current Session Data to Cloud", key="save_session_data_button"):
            save_session_data_to_firestore()
    with cloud_data_cols[1]:
        if st.button("🔄 Load Session Data from Cloud", key="load_session_data_button"):
            load_session_data_from_firestore()
            st.rerun() # Rerun to apply loaded data to the UI

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


# ======================
# Page Routing via function calls (remaining pages)
# ======================
elif tab == "🧠 Resume Screener":
    try:
        # Import the screener page function (assuming it's in a separate file)
        from screener import resume_screener_page
        resume_screener_page() # Call the imported function
        # The logging and pending approval logic here should ideally be handled within resume_screener_page itself
        # after a successful screening operation. For now, keeping it here for demonstration.
        if 'comprehensive_df' in st.session_state and not st.session_state['comprehensive_df'].empty:
            # Log activity only if new data was added to comprehensive_df
            current_df_len = len(st.session_state['comprehensive_df'])
            if st.session_state.get('last_screen_log_count', 0) < current_df_len:
                log_activity(f"Performed resume screening for {current_df_len} candidates.")
                st.session_state.last_screen_log_count = current_df_len

            # Example: Triggering a pending approval for a high-scoring candidate
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
                    log_activity(f"Candidate '{result['Candidate Name']}' sent for approval (high score).")
                    st.toast(f"Candidate {result['Candidate Name']} sent for approval!")

    except ImportError:
        st.error("`screener.py` not found or `resume_screener_page` function not defined. Please ensure 'screener.py' exists and contains the 'resume_screener_page' function.")
    except Exception as e:
        st.error(f"Error loading Resume Screener: {e}")

elif tab == "📁 Manage JDs":
    try:
        with open("manage_jds.py", encoding="utf-8") as f:
            exec(f.read())
    except FileNotFoundError:
        st.info("`manage_jds.py` not found. Please ensure the file exists in the same directory.")
    except Exception as e:
        st.error(f"Error loading Manage JDs: {e}")

elif tab == "📊 Screening Analytics":
    analytics_dashboard_page()

elif tab == "📤 Email Candidates":
    try:
        # Import the email sender function (assuming it's in a separate file)
        from email_sender import send_email_to_candidate
        send_email_to_candidate() # Call the imported function
    except ImportError:
        st.error("`email_sender.py` not found or `send_email_to_candidate` function not defined. Please ensure 'email_sender.py' exists and contains the 'send_email_to_candidate' function.")
    except Exception as e:
        st.error(f"Error loading Email Candidates: {e}")

elif tab == "🔍 Search Resumes":
    try:
        with open("search.py", encoding="utf-8") as f:
            exec(f.read())
    except FileNotFoundError:
        st.info("`search.py` not found. Please ensure the file exists in the same directory.")
    except Exception as e:
        st.error(f"Error loading Search Resumes: {e}")

elif tab == "📝 Candidate Notes":
    try:
        with open("notes.py", encoding="utf-8") as f:
            exec(f.read())
    except FileNotFoundError:
        st.info("`notes.py` not found. Please ensure the file exists in the same directory.")
    except Exception as e:
        st.error(f"Error loading Candidate Notes: {e}")

elif tab == "❓ Feedback & Help":
    # Import the feedback page function
    from feedback import feedback_and_help_page
    if 'user_email' not in st.session_state:
        st.session_state['user_email'] = st.session_state.get('username', 'anonymous_user')
    feedback_and_help_page()

elif tab == "🚪 Logout":
    log_activity(f"User '{st.session_state.get('username', 'anonymous_user')}' logged out.")
    st.session_state.authenticated = False
    st.session_state.pop('username', None)
    st.success("✅ Logged out.")
    st.rerun()
