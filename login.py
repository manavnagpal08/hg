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
    # Regex for a simple email check (covers @ and at least one . after @)
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

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


def login_section():
    """
    Handles user login (HR) and public registration.
    Sets session state variables for authentication and user details.
    Returns True if authenticated, False otherwise.
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

    # If already authenticated, just return True
    if st.session_state.authenticated:
        return True

    # Initialize active_login_tab_selection if not present
    if "active_login_tab_selection" not in st.session_state:
        # Default to 'Register' if no users, otherwise 'Login'
        if not os.path.exists(USER_DB_FILE) or len(load_users()) == 0:
            st.session_state.active_login_tab_selection = "Register"
        else:
            st.session_state.active_login_tab_selection = "Login"

    st.title("🔐 ScreenerPro Access")
    st.markdown("Please select your access mode.")

    # Radio buttons for mode selection
    mode = st.radio(
        "Select Mode:",
        ("HR Mode", "Candidate Mode"),
        key="mode_selection",
        horizontal=True,
        index=0 if st.session_state.app_mode == "hr_mode" else 1 if st.session_state.app_mode == "candidate_mode" else 0 # Default to HR mode initially
    )

    if mode == "HR Mode":
        st.session_state.app_mode = "hr_mode"
        # Use st.radio to simulate tabs if st.tabs() default_index is not supported
        tab_selection = st.radio(
            "Select an option:",
            ("Login", "Register"),
            key="hr_login_register_radio",
            index=0 if st.session_state.active_login_tab_selection == "Login" else 1
        )

        if tab_selection == "Login":
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
                            st.session_state.candidate_logged_in = False # Ensure candidate mode is off
                            st.success("✅ HR Login successful!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password.")
        
        elif tab_selection == "Register":
            register_section()

    elif mode == "Candidate Mode":
        st.session_state.app_mode = "candidate_mode"
        st.subheader("Candidate Login")
        if not st.session_state.candidate_logged_in:
            # Simple password-based login for demonstration
            candidate_password = st.text_input("Enter Candidate Access Code", type="password", key="candidate_access_code")
            if st.button("Login as Candidate"):
                if candidate_password == "candidate123": # Placeholder: Replace with a more secure method
                    st.session_state.candidate_logged_in = True
                    st.session_state.authenticated = False # Candidates are not HR authenticated users
                    st.session_state.username = "Candidate User" # Generic name for candidate mode
                    st.session_state.user_company = "N/A" # No company for candidates
                    st.success("Logged in as Candidate!")
                    st.rerun()
                else:
                    st.error("Incorrect Access Code. Please try again.")
        else:
            st.info("You are already logged in as a Candidate.")
            # Option to logout from candidate mode
            if st.button("Logout from Candidate Mode"):
                st.session_state.candidate_logged_in = False
                st.session_state.app_mode = "select_mode"
                st.session_state.username = None
                st.session_state.user_company = None
                st.rerun()

    return st.session_state.authenticated or st.session_state.candidate_logged_in

# Helper function to check if the current user is an admin
def is_current_user_admin():
    # Check if the current username is in the ADMIN_USERNAME tuple
    return st.session_state.get("authenticated", False) and st.session_state.get("username") in ADMIN_USERNAME

# Example of how to use it if running login.py directly for testing
if __name__ == "__main__":
    st.set_page_config(
        page_title="ScreenerPro Login",
        page_icon="🔐",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    # Apply custom CSS (kept for standalone testing)
    st.markdown(
        """
        <style>
            html, body, [class*="css"] { font-family: 'Inter', sans-serif; transition: background-color 0.3s ease, color 0.3s ease; }
            .main .block-container { padding: 2.5rem; border-radius: 20px; box-shadow: 0 15px 40px rgba(0,0,0,0.15); animation: fadeIn 0.8s ease-in-out; transition: background 0.3s ease, box-shadow 0.3s ease; }
            @keyframes fadeIn { 0% { opacity: 0; transform: translateY(20px); } 100% { opacity: 1; transform: translateY(0); } }
            h1, h2, h3, h4, h5, h6 { color: #00cec9; font-weight: 700; letter-spacing: -0.02em; }
            .stButton>button { background-color: #00cec9; color: white; border-radius: 10px; padding: 0.7rem 1.4rem; font-weight: 600; border: none; transition: all 0.3s ease; box-shadow: 0 5px 15px rgba(0,0,0,0.15); cursor: pointer; letter-spacing: 0.02em; }
            .stButton>button:hover { background-color: #00b0a8; transform: translateY(-4px); box-shadow: 0 8px 20px rgba(0,0,0,0.3); }
            .stRadio div[role="radiogroup"] label { border-radius: 14px; padding: 0.9rem 1.2rem; font-weight: 600; border: none; transition: all 0.2s ease; box-shadow: 0 3px 10px rgba(0,0,0,0.05); width: 100%; text-align: left; cursor: pointer; display: flex; align-items: center; }
            .stRadio div[role="radiogroup"] label span:first-child { display: none; }
            .stRadio div[role="radiogroup"] label:hover { background-color: rgba(0,206,201,0.1); transform: translateX(6px); box-shadow: 0 5px 15px rgba(0,0,0,0.15); }
            .stRadio div[role="radiogroup"] input:checked + div { background-color: #00cec9; color: white; box-shadow: 0 6px 18px rgba(0,0,0,0.25); transform: translateX(0); border: 1px solid #00a09a; }
            .stRadio div[role="radiogroup"] input:checked + div p { color: white; }
            #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Ensure all admin users exist for testing
    users = load_users()
    default_admin_password = "adminpass" # Define a default password for new admin users

    for admin_user in ADMIN_USERNAME:
        if admin_user not in users:
            users[admin_user] = {"password": hash_password(default_admin_password), "status": "active", "company": "AdminCo"}
            st.info(f"Created default admin user: {admin_user} with password '{default_admin_password}'")
    save_users(users) # Save after potentially adding new admin users

    # Call the login section
    logged_in = login_section()

    if logged_in:
        st.markdown("---")
        st.subheader("Login Status (for testing login.py)")
        st.write(f"**Authenticated:** {st.session_state.authenticated}")
        st.write(f"**Candidate Logged In:** {st.session_state.candidate_logged_in}")
        st.write(f"**App Mode:** {st.session_state.app_mode}")
        if st.session_state.username:
            st.write(f"**Username:** {st.session_state.username}")
        if st.session_state.user_company:
            st.write(f"**Company:** {st.session_state.user_company}")
        
        if st.session_state.authenticated and is_current_user_admin():
            st.markdown("---")
            st.header("Admin Test Section (You are admin)")
            admin_registration_section()
            admin_password_reset_section()
            admin_disable_enable_user_section()

            st.subheader("👥 All Registered Users (Admin View):")
            try:
                users_data = load_users()
                if users_data:
                    display_users = []
                    for user, data in users_data.items():
                        hashed_pass = data.get("password", data) if isinstance(data, dict) else data
                        status = data.get("status", "N/A") if isinstance(data, dict) else "N/A"
                        company = data.get("company", "N/A")
                        display_users.append([user, hashed_pass, status, company])
                    st.dataframe(pd.DataFrame(display_users, columns=["Email/Username", "Hashed Password (DO NOT EXPOSE)", "Status", "Company"]), use_container_width=True)
                else:
                    st.info("No users registered yet.")
            except ImportError:
                st.warning("Pandas is not imported. User table cannot be displayed.")
            except Exception as e:
                st.error(f"Error loading user data: {e}")
        elif st.session_state.authenticated: # HR user, not admin
            st.info("You are logged in as an HR user. Admin features are not available.")
        
        if st.session_state.authenticated or st.session_state.candidate_logged_in:
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.candidate_logged_in = False
                st.session_state.app_mode = "select_mode"
                st.session_state.pop('username', None)
                st.session_state.pop('user_company', None)
                st.rerun()
    else:
        st.info("Please login or register to continue.")
