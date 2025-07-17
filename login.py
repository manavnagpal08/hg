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
        # Ensure each user has a 'status', 'company', and 'role' key for backward compatibility
        for username, data in users.items():
            if isinstance(data, str): # Old format: "username": "hashed_password"
                users[username] = {"password": data, "status": "active", "company": "N/A", "role": "HR"}
            elif "status" not in data:
                data["status"] = "active"
            if "company" not in data: # Add company field if missing
                data["company"] = "N/A"
            if "role" not in data: # Add role field if missing, default to "HR"
                data["role"] = "HR"
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

def hr_register_section():
    """HR-specific registration form."""
    st.subheader("📝 Register New HR Account")
    with st.form("hr_registration_form", clear_on_submit=True):
        new_username = st.text_input("Choose Username (Email address)", key="new_username_reg_hr")
        new_company_name = st.text_input("Company Name", key="new_company_name_reg_hr")
        new_password = st.text_input("Choose Password", type="password", key="new_password_reg_hr")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password_reg_hr")
        register_button = st.form_submit_button("Register HR Account")

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
                        "company": new_company_name,
                        "role": "HR" # Assign HR role
                    }
                    save_users(users)
                    st.success("✅ HR Account registered successfully! You can now login.")

def candidate_register_section():
    """Candidate-specific registration form."""
    st.subheader("📝 Register New Candidate Account")
    with st.form("candidate_registration_form", clear_on_submit=True):
        new_username = st.text_input("Choose Username (Email address)", key="new_username_reg_candidate")
        new_password = st.text_input("Choose Password", type="password", key="new_password_reg_candidate")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password_reg_candidate")
        register_button = st.form_submit_button("Register Candidate Account")

        if register_button:
            if not new_username or not new_password or not confirm_password:
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
                        "company": "N/A", # Candidates don't have a company in this context
                        "role": "Candidate" # Assign Candidate role
                    }
                    save_users(users)
                    st.success("✅ Candidate Account registered successfully! You can now login.")


def admin_registration_section():
    """Admin-driven user creation form."""
    st.subheader("➕ Create New User Account (Admin Only)")
    with st.form("admin_registration_form", clear_on_submit=True):
        new_username = st.text_input("New User's Username (Email)", key="new_username_admin_reg")
        new_company_name = st.text_input("New User's Company Name", key="new_company_name_admin_reg") # New field
        new_password = st.text_input("New User's Password", type="password", key="new_password_admin_reg")
        new_user_role = st.selectbox("Assign Role", ["HR", "Candidate"], key="new_user_role_admin_reg") # New role selection
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
                    "company": new_company_name,
                    "role": new_user_role # Store selected role
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

def admin_change_user_role_section():
    """Admin-driven user role change form."""
    st.subheader("🔄 Change User Role (Admin Only)")
    users = load_users()
    user_options = [user for user in users.keys() if user not in ADMIN_USERNAME]

    if not user_options:
        st.info("No other users to change roles for.")
        return

    with st.form("admin_change_user_role_form", clear_on_submit=False):
        selected_user = st.selectbox("Select User to Change Role For", user_options, key="change_role_user_select")
        
        current_role = users[selected_user]["role"]
        st.info(f"Current role of '{selected_user}': **{current_role.upper()}**")

        new_role = st.selectbox("New Role", ["HR", "Candidate"], index=["HR", "Candidate"].index(current_role), key="new_role_select")
        
        if st.form_submit_button(f"Change Role to {new_role.upper()}"):
            users[selected_user]["role"] = new_role
            save_users(users)
            st.success(f"✅ User '{selected_user}' role set to **{new_role.upper()}**.")
            st.rerun()


def login_section():
    """Handles user login for HR and Candidate, and public registration."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "user_role" not in st.session_state: # Initialize user_role
        st.session_state.user_role = None
    
    # If already authenticated, just return True and don't render the form
    if st.session_state.authenticated:
        return True

    # Use st.tabs for main portals: HR and Candidate
    hr_portal_tab, candidate_portal_tab = st.tabs(["HR Portal", "Candidate Portal"])

    with hr_portal_tab:
        login_tab, register_tab = st.tabs(["Login", "Register"])
        with login_tab:
            st.subheader("🔐 HR Login")
            st.info("Login here if you are an HR professional or Administrator.")
            with st.form("hr_login_form", clear_on_submit=False):
                username = st.text_input("HR Username (Email)", key="username_hr_login")
                password = st.text_input("HR Password", type="password", key="password_hr_login")
                submitted = st.form_submit_button("Login as HR")

                if submitted:
                    users = load_users()
                    if username not in users:
                        st.error("❌ Invalid username or password.")
                    else:
                        user_data = users[username]
                        if user_data["status"] == "disabled":
                            st.error("❌ Your HR account has been disabled. Please contact an administrator.")
                        elif not check_password(password, user_data["password"]):
                            st.error("❌ Invalid username or password.")
                        elif user_data.get("role") == "Candidate": # Specific check for HR login
                            st.error("❌ This is a Candidate account. Please use the 'Candidate Portal' tab.")
                        else: # HR or Admin role
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.session_state.user_company = user_data.get("company", "N/A")
                            st.session_state.user_role = user_data.get("role", "HR") # Should be HR or Admin
                            st.success("✅ HR Login successful! Redirecting...")
                            st.rerun()
        with register_tab:
            hr_register_section()

    with candidate_portal_tab:
        login_tab, register_tab = st.tabs(["Login", "Register"])
        with login_tab:
            st.subheader("👤 Candidate Login")
            st.info("Login here if you are a candidate.")
            with st.form("candidate_login_form", clear_on_submit=False):
                username = st.text_input("Candidate Username (Email)", key="username_candidate_login")
                password = st.text_input("Candidate Password", type="password", key="password_candidate_login")
                submitted = st.form_submit_button("Login as Candidate")

                if submitted:
                    users = load_users()
                    if username not in users:
                        st.error("❌ Invalid username or password.")
                    else:
                        user_data = users[username]
                        if user_data["status"] == "disabled":
                            st.error("❌ Your Candidate account has been disabled. Please contact an administrator.")
                        elif not check_password(password, user_data["password"]):
                            st.error("❌ Invalid username or password.")
                        elif user_data.get("role") != "Candidate": # Specific check for Candidate login
                            st.error("❌ This is an HR/Admin account. Please use the 'HR Portal' tab.")
                        else: # Candidate role
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.session_state.user_company = user_data.get("company", "N/A")
                            st.session_state.user_role = "Candidate"
                            st.success("✅ Candidate Login successful! Redirecting...")
                            st.rerun()
        with register_tab:
            candidate_register_section()

    return st.session_state.authenticated

# Helper function to check if the current user is an admin
def is_current_user_admin():
    # Check if the current username is in the ADMIN_USERNAME tuple
    return st.session_state.get("authenticated", False) and st.session_state.get("username") in ADMIN_USERNAME

# Example of how to use it if running login.py directly for testing
if __name__ == "__main__":
    st.set_page_config(page_title="Login/Register", layout="centered")
    st.title("ScreenerPro Authentication (Test)")
    
    # Ensure all admin users exist for testing
    users = load_users()
    default_admin_password = "adminpass" # Define a default password for new admin users

    for admin_user in ADMIN_USERNAME:
        if admin_user not in users:
            users[admin_user] = {"password": hash_password(default_admin_password), "status": "active", "company": "AdminCo", "role": "HR"} # Admins are HR by default
            st.info(f"Created default admin user: {admin_user} with password '{default_admin_password}'")
    
    # Removed automatic creation of default HR and Candidate users
    # if "hr@forscreenerpro" not in users:
    #     users["hr@forscreenerpro"] = {"password": hash_password("hrpass"), "status": "active", "company": "HRTeam", "role": "HR"}
    #     st.info("Created default HR user: hr@forscreenerpro with password 'hrpass'")

    # if "candidate@forscreenerpro" not in users:
    #     users["candidate@forscreenerpro"] = {"password": hash_password("candpass"), "status": "active", "company": "ApplicantCo", "role": "Candidate"}
    #     st.info("Created default Candidate user: candidate@forscreenerpro with password 'candpass'")

    save_users(users) # Save after potentially adding new users

    if login_section():
        st.write(f"Welcome, {st.session_state.username}! Your role is: {st.session_state.user_role}")
        st.write(f"Your company: {st.session_state.get('user_company', 'N/A')}") # Display company
        st.write("You are logged in.")
        
        if is_current_user_admin():
            st.markdown("---")
            st.header("Admin Test Section (You are admin)")
            admin_registration_section()
            admin_password_reset_section()
            admin_disable_enable_user_section()
            admin_change_user_role_section() # New admin function

            st.subheader("👥 All Registered Users (Admin View):")
            # This part requires pandas, which is typically in main.py.
            # For standalone login.py testing, ensure pandas is imported.
            try:
                # import pandas as pd # Already imported at the top of the file
                users_data = load_users()
                if users_data:
                    display_users = []
                    for user, data in users_data.items():
                        hashed_pass = data.get("password", data) if isinstance(data, dict) else data
                        status = data.get("status", "N/A") if isinstance(data, dict) else "N/A"
                        company = data.get("company", "N/A") # Get company
                        role = data.get("role", "N/A") # Get role
                        display_users.append([user, hashed_pass, status, company, role]) # Add company and role to the list
                    st.dataframe(pd.DataFrame(display_users, columns=["Email/Username", "Hashed Password (DO NOT EXPOSE)", "Status", "Company", "Role"]), use_container_width=True) # Update columns list
                else:
                    st.info("No users registered yet.")
            except ImportError:
                st.warning("Pandas is not imported. User table cannot be displayed.")
            except Exception as e:
                st.error(f"Error loading user data: {e}")
        else:
            st.info(f"Log in as one of the admin accounts ({', '.join(ADMIN_USERNAME)}) to see admin features.")
            
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.pop('username', None)
            st.session_state.pop('user_role', None) # Clear role on logout
            st.session_state.pop('user_company', None) # Clear company on logout
            st.rerun()
    else:
        st.info("Please login or register to continue.")
