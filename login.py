import streamlit as st
import json
import bcrypt
import os
import re # Import regex for email validation

USER_DB_FILE = "users.json"
CANDIDATE_DB_FILE = "candidate_users.json" # New file for candidate credentials
ADMIN_USERNAME = ("admin@forscreenerpro", "admin@forscreenerpro2") # Defined in main.py, but good to have a reference

# --- Helper Functions for User Management (Recruiter/Admin) ---
def load_users():
    """Loads user data from the JSON file."""
    if not os.path.exists(USER_DB_FILE):
        return {}
    with open(USER_DB_FILE, "r") as f:
        return json.load(f)

def save_users(users_data):
    """Saves user data to the JSON file."""
    with open(USER_DB_FILE, "w") as f:
        json.dump(users_data, f, indent=4)

def hash_password(password):
    """Hashes a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed_password):
    """Checks if a password matches a hashed password."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def is_valid_email(email):
    """Validates an email address format."""
    # Basic regex for email validation
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def is_current_user_admin():
    """Checks if the currently logged-in user is an admin."""
    return st.session_state.get('username') in ADMIN_USERNAME

def is_current_user_candidate():
    """Checks if the currently logged-in user is a candidate."""
    return st.session_state.get('user_type') == 'candidate'

# --- Candidate User Management Functions ---
def load_candidate_users():
    """Loads candidate user data from the JSON file."""
    if not os.path.exists(CANDIDATE_DB_FILE):
        return {}
    with open(CANDIDATE_DB_FILE, "r") as f:
        return json.load(f)

def save_candidate_users(users_data):
    """Saves candidate user data to the JSON file."""
    with open(CANDIDATE_DB_FILE, "w") as f:
        json.dump(users_data, f, indent=4)

def candidate_registration_section():
    """Streamlit section for candidate registration."""
    st.subheader("📝 Candidate Registration")
    with st.form("candidate_registration_form", clear_on_submit=True):
        new_candidate_email = st.text_input("Candidate Email", key="new_candidate_email_reg")
        new_candidate_password = st.text_input("Password", type="password", key="new_candidate_password_reg")
        confirm_candidate_password = st.text_input("Confirm Password", type="password", key="confirm_candidate_password_reg")
        
        submitted = st.form_submit_button("Register Candidate Account")

        if submitted:
            if not new_candidate_email or not new_candidate_password or not confirm_candidate_password:
                st.error("All fields are required.")
            elif not is_valid_email(new_candidate_email):
                st.error("Please enter a valid email address.")
            elif new_candidate_password != confirm_candidate_password:
                st.error("Passwords do not match.")
            else:
                candidate_users = load_candidate_users()
                if new_candidate_email in candidate_users:
                    st.error("Candidate with this email already exists.")
                else:
                    hashed_password = hash_password(new_candidate_password)
                    candidate_users[new_candidate_email] = {
                        "password": hashed_password,
                        "status": "active",
                        "registration_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    save_candidate_users(candidate_users)
                    st.success("✅ Candidate account registered successfully! You can now log in.")

# --- Login Sections (Modified to include Candidate Login) ---
def login_section():
    """
    Handles the login process for both recruiter/admin and candidates.
    Returns True if authenticated, False otherwise.
    """
    if st.session_state.get('authenticated', False):
        return True

    st.sidebar.empty() # Clear sidebar content before login form

    login_type = st.radio(
        "Select Login Type",
        ["Recruiter/Admin Login", "Candidate Login"],
        key="login_type_selector"
    )

    if login_type == "Recruiter/Admin Login":
        st.subheader("🔐 Recruiter/Admin Login")
        username = st.text_input("Username (Email)", key="recruiter_username")
        password = st.text_input("Password", type="password", key="recruiter_password")
        login_button = st.button("Login as Recruiter/Admin")

        if login_button:
            users = load_users()
            if username in users and check_password(password, users[username].get("password", users[username])):
                if users[username].get("status", "active") == "active":
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.user_type = 'recruiter_admin'
                    st.success(f"Welcome, {username}!")
                    st.rerun()
                else:
                    st.error("Your account is disabled. Please contact an administrator.")
            else:
                st.error("Invalid Username or Password.")
    else: # Candidate Login
        st.subheader("👤 Candidate Login")
        candidate_email = st.text_input("Email", key="candidate_email_login")
        candidate_password = st.text_input("Password", type="password", key="candidate_password_login")
        login_button = st.button("Login as Candidate")

        if login_button:
            candidate_users = load_candidate_users()
            if candidate_email in candidate_users and check_password(candidate_password, candidate_users[candidate_email].get("password", candidate_users[candidate_email])):
                if candidate_users[candidate_email].get("status", "active") == "active":
                    st.session_state.authenticated = True
                    st.session_state.username = candidate_email
                    st.session_state.user_type = 'candidate'
                    st.success(f"Welcome, {candidate_email}!")
                    st.rerun()
                else:
                    st.error("Your candidate account is disabled. Please contact support.")
            else:
                st.error("Invalid Email or Password.")
        
        st.markdown("---")
        st.info("Don't have a candidate account? Register below.")
        candidate_registration_section() # Offer registration directly on the candidate login page

    return st.session_state.get('authenticated', False)


# --- Admin Tools Functions (Recruiter/Admin only) ---
def admin_registration_section():
    """Streamlit section for admin to register new users."""
    st.subheader("➕ Register New Recruiter/Admin User")
    with st.form("new_user_registration_form", clear_on_submit=True):
        new_username = st.text_input("New Username (Email)", key="new_username_reg")
        new_password = st.text_input("New Password", type="password", key="new_password_reg")
        confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_password_reg")
        company_name = st.text_input("Company Name (Optional)", key="company_name_reg")
        
        submitted = st.form_submit_button("Register User")

        if submitted:
            if not new_username or not new_password or not confirm_password:
                st.error("Username, Password, and Confirm Password are required.")
            elif not is_valid_email(new_username):
                st.error("Please enter a valid email address for the username.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                users = load_users()
                if new_username in users:
                    st.error("User with this username already exists.")
                else:
                    hashed_password = hash_password(new_password)
                    users[new_username] = {
                        "password": hashed_password,
                        "status": "active",
                        "company": company_name # Store company name
                    }
                    save_users(users)
                    st.success("✅ User registered successfully!")

def admin_password_reset_section():
    """Streamlit section for admin to reset user passwords."""
    st.subheader("🔑 Reset User Password")
    users = load_users()
    user_options = list(users.keys())

    if not user_options:
        st.info("No users to reset passwords for.")
        return

    with st.form("password_reset_form", clear_on_submit=True):
        user_to_reset = st.selectbox("Select User to Reset Password", user_options, key="user_to_reset_pwd")
        new_password = st.text_input("New Password", type="password", key="new_pwd_reset")
        confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_pwd_reset")
        
        submitted = st.form_submit_button("Reset Password")

        if submitted:
            if not new_password or not confirm_password:
                st.error("New Password and Confirm Password are required.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                users[user_to_reset]["password"] = hash_password(new_password)
                save_users(users)
                st.success(f"✅ Password for {user_to_reset} reset successfully!")

def admin_disable_enable_user_section():
    """Streamlit section for admin to disable/enable users."""
    st.subheader("🚫 Enable/Disable User Account")
    users = load_users()
    user_options = list(users.keys())

    if not user_options:
        st.info("No users to manage.")
        return

    with st.form("disable_enable_user_form", clear_on_submit=True):
        user_to_manage = st.selectbox("Select User Account", user_options, key="user_to_manage_status")
        current_status = users[user_to_manage].get("status", "active")
        st.write(f"Current Status: **{current_status.upper()}**")
        
        action = st.radio("Action", ["Activate", "Deactivate"], index=0 if current_status == "active" else 1, key="user_status_action")
        
        submitted = st.form_submit_button(f"{action} User")

        if submitted:
            new_status = "active" if action == "Activate" else "disabled"
            users[user_to_manage]["status"] = new_status
            save_users(users)
            st.success(f"✅ User {user_to_manage} account has been {new_status}!")

