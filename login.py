import streamlit as st
import json
import bcrypt
import os
import re
import pandas as pd # Needed for admin view, even if not directly used in core auth logic

# File to store user credentials
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
        # Ensure each user has a 'status' key, 'company' key, and 'role' key for backward compatibility
        for username, data in users.items():
            if isinstance(data, str): # Old format: "username": "hashed_password"
                # Default role for old format users will be 'hr'
                users[username] = {"password": data, "status": "active", "company": "N/A", "role": "hr"}
            elif "status" not in data:
                data["status"] = "active"
            if "company" not in data:
                data["company"] = "N/A"
            if "role" not in data: # Assign a default role if missing
                # Assign 'admin' role if it's an ADMIN_USERNAME, otherwise 'hr' (assuming public register was always for HR)
                data["role"] = "admin" if username in ADMIN_USERNAME else "hr"
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

def register_user(username, company_name, password, role):
    """Registers a new user with a specified role."""
    users = load_users()
    if not username or not password or not company_name:
        st.error("Please fill in all fields.")
        return False
    elif not is_valid_email(username):
        st.error("Please enter a valid email address for the username.")
        return False
    elif username in users:
        st.error("Username already exists. Please choose a different one.")
        return False
    else:
        # Prevent public registration of admin accounts
        if username in ADMIN_USERNAME and role != "admin":
            st.error("Cannot register an admin account through the public portal. Please contact an administrator.")
            return False

        users[username] = {
            "password": hash_password(password),
            "status": "active",
            "company": company_name,
            "role": role # Store the user's role
        }
        save_users(users)
        st.success(f"✅ Registration successful for {role.upper()} account! You can now switch to the 'Login' option.")
        return True

def register_section_ui(portal_type):
    """Public self-registration form for HR or Candidate."""
    st.subheader(f"📝 Create New {portal_type.replace('_portal', '').upper()} Account")
    # Determine the role based on the portal type
    assigned_role = "hr" if portal_type == "hr_portal" else "candidate" # Use internal role names

    with st.form(f"registration_form_{portal_type}", clear_on_submit=True):
        new_username = st.text_input("Choose Username (Email address required)", key=f"new_username_reg_{portal_type}")
        new_company_label = "Company Name" if portal_type == "hr_portal" else "Your Name / Organization"
        new_company_name = st.text_input(new_company_label, key=f"new_company_name_reg_{portal_type}")
        new_password = st.text_input("Choose Password", type="password", key=f"new_password_reg_{portal_type}")
        confirm_password = st.text_input("Confirm Password", type="password", key=f"confirm_password_reg_{portal_type}")
        register_button = st.form_submit_button("Register New Account")

        if register_button:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success = register_user(new_username, new_company_name, new_password, assigned_role)
                if success:
                    # Manually set the session state to switch to Login option
                    st.session_state[f"active_login_tab_selection_{portal_type}"] = "Login"
                    st.rerun() # Rerun to update the radio button selection

def admin_registration_section():
    """Admin-driven user creation form."""
    st.subheader("➕ Create New User Account (Admin Only)")
    st.info("Admin-created users will be assigned the 'HR' role by default.")
    with st.form("admin_registration_form", clear_on_submit=True):
        new_username = st.text_input("New User's Username (Email)", key="new_username_admin_reg")
        new_company_name = st.text_input("New User's Company Name", key="new_company_name_admin_reg")
        new_password = st.text_input("New User's Password", type="password", key="new_password_admin_reg")
        # No role selection here; assumed 'hr' for admin-created users
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
                    "company": new_company_name,
                    "role": "hr" # Admin created users are 'hr' by default
                }
                save_users(users)
                st.success(f"✅ User '{new_username}' (HR role) added successfully!")


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
        new_password = st.text_input("New Password", type="password", key="new_password_reset")
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

        if st.form_submit_button(f"Toggle to {'Disable' if current_status == "active" else "Enable"} User"):
            new_status = "disabled" if current_status == "active" else "active"
            users[selected_user]["status"] = new_status
            save_users(users)
            st.success(f"✅ User '{selected_user}' status set to **{new_status.upper()}**.")
            st.rerun() # Rerun to update the displayed status immediately


def login_user(username, password, expected_portal_type):
    """
    Authenticates a user and checks their role against the expected portal type.
    expected_portal_type should be "hr_portal" or "candidate_portal".
    Returns True if authenticated and role matches, False otherwise.
    """
    users = load_users()
    if username not in users:
        st.error("❌ Invalid username or password.")
        return False
    
    user_data = users[username]
    
    # Map portal type to internal role(s) allowed
    if expected_portal_type == "hr_portal":
        required_roles = ["hr", "admin"] # HR portal can be accessed by HR or Admin
    elif expected_portal_type == "candidate_portal":
        required_roles = ["candidate"] # Candidate portal only for candidates
    else:
        st.error("Invalid portal type specified for login.")
        return False

    if user_data["status"] == "disabled":
        st.error("❌ Your account has been disabled. Please contact an administrator.")
        return False
    
    if check_password(password, user_data["password"]):
        # Check if the user's role matches the allowed roles for this portal
        if user_data.get("role") in required_roles:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.user_company = user_data.get("company", "N/A")
            st.session_state.user_role = user_data.get("role") # Store the authenticated user's role
            st.success("✅ Login successful!")
            return True
        else:
            st.error(f"❌ This account is not authorized for the {expected_portal_type.replace('_', ' ').upper()}.")
            return False
    else:
        st.error("❌ Invalid username or password.")
        return False

def login_section_ui(portal_type, load_session_data_callback=None):
    """
    Displays the login and register UI for a specific portal type (e.g., "hr_portal", "candidate_portal").
    Returns True if user successfully logs in for the given portal, False otherwise.
    """
    # Initialize relevant session state variables if not present
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "user_role" not in st.session_state:
        st.session_state.user_role = None
    
    # Initialize active_login_tab_selection for this specific portal if not present
    tab_key = f"active_login_tab_selection_{portal_type}"
    if tab_key not in st.session_state:
        st.session_state[tab_key] = "Login" 

    # Determine the required role(s) for this portal
    if portal_type == "hr_portal":
        allowed_roles = ["hr", "admin"]
    elif portal_type == "candidate_portal":
        allowed_roles = ["candidate"]
    else:
        allowed_roles = [] # Should not happen

    # If already authenticated AND role matches the current portal, skip login forms
    if st.session_state.authenticated and st.session_state.get("user_role") in allowed_roles:
        return True 

    # Use st.radio to simulate tabs for Login/Register within each portal
    tab_selection = st.radio(
        f"Choose an option for the {portal_type.replace('_', ' ').upper()}:",
        ("Login", "Register"),
        key=f"login_register_radio_{portal_type}", # Unique key for each portal's radio button
        index=0 if st.session_state[tab_key] == "Login" else 1
    )

    if tab_selection == "Login":
        st.subheader(f"🔐 {portal_type.replace('_', ' ').upper()} Login")
        st.info(f"If you don't have an account, please go to the 'Register' option first for the {portal_type.replace('_', ' ').upper()}.")
        with st.form(f"login_form_{portal_type}", clear_on_submit=False):
            username = st.text_input("Username", key=f"username_login_{portal_type}")
            password = st.text_input("Password", type="password", key=f"password_login_{portal_type}")
            submitted = st.form_submit_button("Login")

            if submitted:
                # Call the specific login_user function
                if login_user(username, password, portal_type):
                    # If login is successful, call the provided callback to load session data
                    if load_session_data_callback:
                        load_session_data_callback(st.session_state.username)
                    st.rerun() # Rerun immediately after successful login
                    return True 
                else:
                    return False # Login failed

    elif tab_selection == "Register":
        register_section_ui(portal_type)
    
    return False # Not yet authenticated or role doesn't match

# Helper function to check if the current user is an admin
def is_current_user_admin():
    return st.session_state.get("authenticated", False) and st.session_state.get("user_role") == "admin"

# Helper function to check if the current user is an HR (includes admins)
def is_current_user_hr():
    return st.session_state.get("authenticated", False) and \
           (st.session_state.get("user_role") == "hr" or st.session_state.get("user_role") == "admin")

# Helper function to check if the current user is a Candidate
def is_current_user_candidate():
    return st.session_state.get("authenticated", False) and st.session_state.get("user_role") == "candidate"

