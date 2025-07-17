import streamlit as st
import json
import bcrypt
import os
import re # Import regex for email validation
import pandas as pd # Ensure pandas is imported for DataFrame display

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
                # Assign 'hr' role to existing non-admin users for consistency with HR Portal
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
    # Regex for a simple email check (covers @ and at least one . after @)
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

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
        users[username] = {
            "password": hash_password(password),
            "status": "active",
            "company": company_name,
            "role": role # Store the user's role
        }
        save_users(users)
        st.success(f"✅ Registration successful for {role} account! You can now switch to the 'Login' option.")
        return True

def register_section(portal_type):
    """Public self-registration form for HR or Candidate."""
    st.subheader(f"📝 Create New {portal_type.upper()} Account")
    with st.form(f"registration_form_{portal_type}", clear_on_submit=True):
        new_username = st.text_input("Choose Username (Email address required)", key=f"new_username_reg_{portal_type}")
        new_company_name = st.text_input("Company Name" if portal_type == "hr" else "Your Name / Organization", key=f"new_company_name_reg_{portal_type}")
        new_password = st.text_input("Choose Password", type="password", key=f"new_password_reg_{portal_type}")
        confirm_password = st.text_input("Confirm Password", type="password", key=f"confirm_password_reg_{portal_type}")
        register_button = st.form_submit_button("Register New Account")

        if register_button:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success = register_user(new_username, new_company_name, new_password, portal_type)
                if success:
                    # Manually set the session state to switch to Login option
                    st.session_state.active_login_tab_selection = "Login"
                    st.rerun() # Rerun to update the radio button selection

def admin_registration_section():
    """Admin-driven user creation form."""
    st.subheader("➕ Create New User Account (Admin Only)")
    with st.form("admin_registration_form", clear_on_submit=True):
        new_username = st.text_input("New User's Username (Email)", key="new_username_admin_reg")
        new_company_name = st.text_input("New User's Company Name", key="new_company_name_admin_reg")
        new_password = st.text_input("New User's Password", type="password", key="new_password_admin_reg")
        new_user_role = st.selectbox("Assign Role", ["hr", "candidate"], key="new_user_role_admin_reg") # Admin can assign role
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
                    "role": new_user_role # Admin sets the role
                }
                save_users(users)
                st.success(f"✅ User '{new_username}' ({new_user_role}) added successfully!")


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

        if st.form_submit_button(f"Toggle to {'Disable' if current_status == 'active' else 'Enable'} User"):
            new_status = "disabled" if current_status == "active" else "active"
            users[selected_user]["status"] = new_status
            save_users(users)
            st.success(f"✅ User '{selected_user}' status set to **{new_status.upper()}**.")
            st.rerun() # Rerun to update the displayed status immediately


def login_user(username, password, expected_role):
    """
    Authenticates a user and checks their role.
    Returns True if authenticated and role matches, False otherwise.
    """
    users = load_users()
    if username not in users:
        st.error("❌ Invalid username or password.")
        return False
    
    user_data = users[username]
    if user_data["status"] == "disabled":
        st.error("❌ Your account has been disabled. Please contact an administrator.")
        return False
    
    if check_password(password, user_data["password"]):
        # Check if the user's role matches the expected portal role
        if user_data.get("role") == expected_role or (expected_role == "hr" and username in ADMIN_USERNAME):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.user_company = user_data.get("company", "N/A")
            st.session_state.user_role = user_data.get("role") # Store the authenticated user's role
            st.success("✅ Login successful!")
            st.rerun()
            return True
        else:
            st.error(f"❌ This account is not authorized for the {expected_role.upper()} portal.")
            return False
    else:
        st.error("❌ Invalid username or password.")
        return False

def login_section_ui(portal_type):
    """
    Displays the login and register UI for a specific portal type (HR or Candidate).
    Returns True if user successfully logs in for the given portal, False otherwise.
    """
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "user_role" not in st.session_state: # Initialize user_role
        st.session_state.user_role = None
    
    # Initialize active_login_tab_selection based on the portal type for unique keys
    tab_key = f"login_register_radio_{portal_type}"
    if tab_key not in st.session_state:
        st.session_state[tab_key] = "Login" # Default to Login

    # If already authenticated and role matches, no need to show login forms
    if st.session_state.authenticated and (st.session_state.user_role == portal_type or \
                                           (portal_type == "hr" and st.session_state.username in ADMIN_USERNAME)):
        return True # User is already logged in and has the correct role for this portal

    # Use st.radio to simulate tabs
    tab_selection = st.radio(
        f"Select an option for {portal_type.upper()} Portal:",
        ("Login", "Register"),
        key=tab_key,
        index=0 if st.session_state[tab_key] == "Login" else 1
    )

    if tab_selection == "Login":
        st.subheader(f"🔐 {portal_type.upper()} Login")
        st.info(f"If you don't have an account, please go to the 'Register' option first for the {portal_type.upper()} Portal.")
        with st.form(f"login_form_{portal_type}", clear_on_submit=False):
            username = st.text_input("Username", key=f"username_login_{portal_type}")
            password = st.text_input("Password", type="password", key=f"password_login_{portal_type}")
            submitted = st.form_submit_button("Login")

            if submitted:
                # Call the specific login_user function
                return login_user(username, password, portal_type)
    
    elif tab_selection == "Register":
        register_section(portal_type)
    
    return False # Not yet authenticated or role doesn't match


# Helper function to check if the current user is an admin
def is_current_user_admin():
    return st.session_state.get("authenticated", False) and st.session_state.get("username") in ADMIN_USERNAME

# Helper function to check if the current user is an HR
def is_current_user_hr():
    return st.session_state.get("authenticated", False) and \
           (st.session_state.get("user_role") == "hr" or st.session_state.get("username") in ADMIN_USERNAME)

# Helper function to check if the current user is a Candidate
def is_current_user_candidate():
    return st.session_state.get("authenticated", False) and st.session_state.get("user_role") == "candidate"

# Example of how to use it if running login.py directly for testing
if __name__ == "__main__":
    st.set_page_config(page_title="Login/Register", layout="centered")
    st.title("ScreenerPro Authentication (Test)")
    
    # Ensure all admin users exist for testing
    users = load_users()
    default_admin_password = "adminpass" # Define a default password for new admin users

    for admin_user in ADMIN_USERNAME:
        if admin_user not in users or users[admin_user].get("role") != "admin": # Ensure admin role is set
            users[admin_user] = {"password": hash_password(default_admin_password), "status": "active", "company": "AdminCo", "role": "admin"}
            st.info(f"Created/Updated default admin user: {admin_user} with password '{default_admin_password}'")
    save_users(users) # Save after potentially adding new admin users

    st.subheader("Select your portal:")
    portal_choice = st.radio("Choose Portal", ["HR Portal", "Candidate Portal"], key="main_portal_choice")

    if portal_choice == "HR Portal":
        if login_section_ui("hr"): # Pass the expected role 'hr'
            st.write(f"Welcome, {st.session_state.username} to the HR Portal!")
            st.write(f"Your company: {st.session_state.get('user_company', 'N/A')}")
            st.write(f"Your role: {st.session_state.get('user_role', 'N/A').upper()}") # Display role
            st.write("You are logged in.")
            
            if is_current_user_admin():
                st.markdown("---")
                st.header("Admin Test Section")
                admin_registration_section()
                admin_password_reset_section()
                admin_disable_enable_user_section()

                st.subheader("👥 All Registered Users (Admin View):")
                try:
                    users_data = load_users()
                    if users_data:
                        display_users = []
                        for user, data in users_data.items():
                            hashed_pass = data.get("password")
                            status = data.get("status", "N/A")
                            company = data.get("company", "N/A")
                            role = data.get("role", "N/A") # Get role
                            display_users.append([user, hashed_pass, status, company, role]) # Add role
                        st.dataframe(pd.DataFrame(display_users, columns=["Email/Username", "Hashed Password (DO NOT EXPOSE)", "Status", "Company", "Role"]), use_container_width=True) # Update columns
                    else:
                        st.info("No users registered yet.")
                except Exception as e:
                    st.error(f"Error loading user data: {e}")
            elif is_current_user_hr():
                st.info("You are logged in as HR. You would see all HR-related pages here.")
            else:
                st.error("You are logged in but do not have HR privileges.")

            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.pop('username', None)
                st.session_state.pop('user_company', None)
                st.session_state.pop('user_role', None)
                st.rerun()
        else:
            st.info("Please login or register for the HR Portal to continue.")

    elif portal_choice == "Candidate Portal":
        if login_section_ui("candidate"): # Pass the expected role 'candidate'
            st.write(f"Welcome, {st.session_state.username} to the Candidate Portal!")
            st.write(f"Your name/organization: {st.session_state.get('user_company', 'N/A')}")
            st.write(f"Your role: {st.session_state.get('user_role', 'N/A').upper()}") # Display role
            st.write("You are logged in.")
            st.subheader("📄 Resume Screener")
            st.write("This is the only page a candidate can see.")
            # Your Resume Screener specific content goes here
            st.info("As a candidate, you only have access to the Resume Screener.")

            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.pop('username', None)
                st.session_state.pop('user_company', None)
                st.session_state.pop('user_role', None)
                st.rerun()
        else:
            st.info("Please login or register for the Candidate Portal to continue.")
