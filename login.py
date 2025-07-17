import streamlit as st
import json
import bcrypt
import os
import re # Import regex for email validation
import pandas as pd # Ensure pandas is imported for DataFrame display
import base64 # For embedding images in certificate simulation

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
        # Ensure each user has 'status', 'company', and 'role' keys for backward compatibility
        for username, data in users.items():
            if isinstance(data, str): # Old format: "username": "hashed_password"
                users[username] = {"password": data, "status": "active", "company": "N/A", "role": "hr"} # Default to HR
            elif "status" not in data:
                data["status"] = "active"
            if "company" not in data: # Add company field if missing
                data["company"] = "N/A"
            if "role" not in data: # Add role field if missing
                data["role"] = "hr" # Default role
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

def register_user(username, password, company_name, role="hr"):
    """Registers a new user (HR or Candidate)."""
    if not username or not password or not company_name:
        st.error("Please fill in all fields.")
        return False
    if not is_valid_email(username):
        st.error("Please enter a valid email address for the username.")
        return False

    users = load_users()
    if username in users:
        st.error("Username already exists. Please choose a different one.")
        return False
    else:
        users[username] = {
            "password": hash_password(password),
            "status": "active",
            "company": company_name,
            "role": role # Store the role
        }
        save_users(users)
        st.success(f"✅ Registration successful for {username}! You can now log in.")
        return True

def register_section(role="hr"):
    """Public self-registration form for HR or Candidate."""
    st.subheader(f"📝 Create New {role.upper()} Account")
    with st.form(f"registration_form_{role}", clear_on_submit=True):
        new_username = st.text_input("Choose Username (Email address required)", key=f"new_username_reg_public_{role}")
        new_company_name = st.text_input("Company Name", key=f"new_company_name_reg_public_{role}") # New field
        new_password = st.text_input("Choose Password", type="password", key=f"new_password_reg_public_{role}")
        confirm_password = st.text_input("Confirm Password", type="password", key=f"confirm_password_reg_public_{role}")
        register_button = st.form_submit_button("Register New Account")

        if register_button:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            elif register_user(new_username, new_password, new_company_name, role):
                # Manually set the session state to switch to Login option
                if role == "hr":
                    st.session_state.active_hr_tab_selection = "Login"
                else: # candidate
                    # After successful candidate registration, switch to login view
                    st.session_state.candidate_view_mode = "login"
                    st.rerun() # Rerun to show the login form

def admin_registration_section():
    """Admin-driven user creation form."""
    st.subheader("➕ Create New User Account (Admin Only)")
    with st.form("admin_registration_form", clear_on_submit=True):
        new_username = st.text_input("New User's Username (Email)", key="new_username_admin_reg")
        new_company_name = st.text_input("New User's Company Name", key="new_company_name_admin_reg") # New field
        new_password = st.text_input("New User's Password", type="password", key="new_password_admin_reg")
        new_user_role = st.selectbox("Assign Role", ["hr", "candidate"], key="new_user_role_admin_reg") # Admin can assign role
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
                    "role": new_user_role
                }
                save_users(users)
                st.success(f"✅ User '{new_username}' ({new_user_role.upper()}) added successfully!")

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

def hr_login_section():
    """Handles HR user login."""
    if "hr_authenticated" not in st.session_state:
        st.session_state.hr_authenticated = False
    if "hr_username" not in st.session_state:
        st.session_state.hr_username = None
    
    # Initialize active_hr_tab_selection if not present
    if "active_hr_tab_selection" not in st.session_state:
        # Default to 'Register' if no users, otherwise 'Login'
        if not os.path.exists(USER_DB_FILE) or len(load_users()) == 0:
            st.session_state.active_hr_tab_selection = "Register"
        else:
            st.session_state.active_hr_tab_selection = "Login"


    if st.session_state.hr_authenticated:
        return True

    tab_selection = st.radio(
        "Select an option:",
        ("Login", "Register"),
        key="hr_login_register_radio",
        index=0 if st.session_state.active_hr_tab_selection == "Login" else 1
    )

    if tab_selection == "Login":
        st.subheader("🔐 HR Login")
        st.info("If you don't have an account, please go to the 'Register' option first.")
        with st.form("hr_login_form", clear_on_submit=False):
            username = st.text_input("Username", key="hr_username_login")
            password = st.text_input("Password", type="password", key="hr_password_login")
            submitted = st.form_submit_button("Login")

            if submitted:
                users = load_users()
                if username not in users:
                    st.error("❌ Invalid username or password. Please register if you don't have an account.")
                else:
                    user_data = users[username]
                    if user_data["status"] == "disabled":
                        st.error("❌ Your account has been disabled. Please contact an administrator.")
                    elif user_data["role"] != "hr" and username not in ADMIN_USERNAME: # Only HR or Admin can login here
                        st.error("❌ This login is for HR accounts only. Please use the Candidate Portal if you are a candidate.")
                    elif check_password(password, user_data["password"]):
                        st.session_state.hr_authenticated = True
                        st.session_state.hr_username = username
                        st.session_state.hr_user_company = user_data.get("company", "N/A")
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password.")
    
    elif tab_selection == "Register":
        register_section(role="hr")

    return st.session_state.hr_authenticated

def generate_certificate_image(name, score):
    """
    Simulates generating a certificate image as a base64 string.
    In a real app, this would be a PDF generation library.
    """
    # Simple SVG for demonstration
    svg_template = f"""
    <svg width="600" height="400" xmlns="http://www.w3.org/2000/svg">
        <rect x="0" y="0" width="600" height="400" fill="#f0f9ff" stroke="#007bff" stroke-width="5" rx="20" ry="20"/>
        <text x="300" y="80" font-family="Arial, sans-serif" font-size="36" fill="#007bff" text-anchor="middle" font-weight="bold">Screener Pro</text>
        <text x="300" y="140" font-family="Arial, sans-serif" font-size="28" fill="#333" text-anchor="middle">Certificate of Achievement</text>
        <text x="300" y="200" font-family="Arial, sans-serif" font-size="24" fill="#555" text-anchor="middle">This certifies that</text>
        <text x="300" y="250" font-family="Arial, sans-serif" font-size="32" fill="#000" text-anchor="middle" font-weight="bold">{name}</text>
        <text x="300" y="300" font-family="Arial, sans-serif" font-size="24" fill="#555" text-anchor="middle">has achieved a screening score of</text>
        <text x="300" y="350" font-family="Arial, sans-serif" font-size="48" fill="#28a745" text-anchor="middle" font-weight="bold">{score}/100</text>
        <text x="500" y="380" font-family="Arial, sans-serif" font-size="14" fill="#777" text-anchor="end">Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}</text>
    </svg>
    """
    b64 = base64.b64encode(svg_template.encode('utf-8')).decode('utf-8')
    return f"data:image/svg+xml;base64,{b64}"

def candidate_portal_content():
    """Content for the candidate portal after login."""
    st.header(f"Welcome, {st.session_state.candidate_username}!")
    st.write("Upload your resume to get an AI-powered screening score and feedback.")

    # Resume Upload
    st.subheader("📄 Resume Upload")
    uploaded_file = st.file_uploader("Drag and drop your resume here, or click to browse", type=["pdf", "doc", "docx"], key="resume_uploader")

    if uploaded_file is not None:
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        st.session_state.uploaded_resume_name = uploaded_file.name
        st.session_state.screening_score = None # Reset score on new upload
        st.session_state.screening_feedback = None
        st.session_state.show_candidate_certificate = False

    # AI Screening
    if st.button("🚀 Get AI Screening Score", key="get_ai_score_button", disabled=(uploaded_file is None or st.session_state.get('is_screening', False))):
        st.session_state.is_screening = True
        st.info("Analyzing your resume... This may take a few moments.")
        
        # Simulate AI processing
        import time
        time.sleep(3) # Simulate network delay/processing
        
        simulated_score = int(st.session_state.get('last_simulated_score', 0)) # Get last score if available
        if simulated_score == 0: # First time or reset
            simulated_score = st.session_state.candidate_username.__hash__() % 41 + 60 # Score between 60-100 based on username hash
        else: # Subsequent screenings, vary slightly
            simulated_score = max(60, min(100, simulated_score + (st.session_state.candidate_username.__hash__() % 11 - 5))) # +/- 5
        
        st.session_state.last_simulated_score = simulated_score

        simulated_strengths = []
        simulated_weaknesses = []

        if simulated_score >= 90:
            simulated_strengths = [
                "Exceptional technical skills and project outcomes.",
                "Strong alignment with industry best practices.",
                "Excellent formatting and readability."
            ]
            simulated_weaknesses = ["Consider adding more quantifiable achievements."]
        elif simulated_score >= 80:
            simulated_strengths = [
                "Strong technical stack: Python, SQL, ML projects.",
                "Good experience and education background.",
                "Well-structured resume."
            ]
            simulated_weaknesses = ["Add clearer project outcomes and metrics.", "Elaborate more on leadership roles."]
        elif simulated_score >= 70:
            simulated_strengths = [
                "Solid foundational skills.",
                "Relevant work experience."
            ]
            simulated_weaknesses = ["Improve project descriptions with outcomes.", "Standardize formatting for consistency.", "Gain more practical experience."]
        else:
            simulated_strengths = [
                "Basic understanding of core concepts."
            ]
            simulated_weaknesses = ["Needs significant improvement in project details.", "Review formatting and grammar.", "Gain more practical experience."]

        st.session_state.screening_score = simulated_score
        st.session_state.screening_feedback = {
            "strengths": simulated_strengths,
            "weaknesses": simulated_weaknesses
        }
        st.session_state.show_candidate_certificate = (simulated_score >= 80)
        st.session_state.is_screening = False
        st.rerun() # Rerun to display results

    # Display Score and Feedback
    if st.session_state.screening_score is not None:
        st.subheader(f"📊 Your Screening Score: {st.session_state.screening_score}/100")

        st.subheader("AI Feedback:")
        st.markdown("---")
        st.markdown("#### ✅ Strengths:")
        for s in st.session_state.screening_feedback["strengths"]:
            st.success(f"- {s}")
        
        st.markdown("#### ❗ Weaknesses:")
        for w in st.session_state.screening_feedback["weaknesses"]:
            st.warning(f"- {w}")
        st.markdown("---")

        # Certificate Display
        if st.session_state.show_candidate_certificate:
            st.subheader("🏆 Your Certificate!")
            certificate_img_data = generate_certificate_image(st.session_state.candidate_username, st.session_state.screening_score)
            st.image(certificate_img_data, caption="Screener Pro Certificate", use_column_width=True)

            st.markdown("#### Share Your Achievement:")
            col1, col2, col3 = st.columns(3)
            
            linkedin_share_url = f"https://www.linkedin.com/shareArticle?mini=true&url=https://www.screenerpro.com&title=Screener%20Pro%20Certificate&summary=I%20just%20scored%20{st.session_state.screening_score}/100%20on%20my%20resume%20screening%20with%20Screener%20Pro!%20Check%20it%20out!%20%23ScreenerPro%20%23ResumeTips%20%23JobSearch&source=ScreenerPro"
            whatsapp_share_text = f"I just scored {st.session_state.screening_score}/100 on my resume screening with Screener Pro! Check it out: https://www.screenerpro.com"

            with col1:
                st.markdown(f"[Share on LinkedIn]({linkedin_share_url})", unsafe_allow_html=True)
            with col2:
                st.markdown(f"[Share on WhatsApp](https://wa.me/?text={whatsapp_share_text})", unsafe_allow_html=True)
            with col3:
                # Simulate badge download (in a real app, this would serve a file)
                st.download_button(
                    label="Download ScreenerPro Badge",
                    data=b"Simulated badge image data", # Replace with actual image data
                    file_name="screenerpro_badge.png",
                    mime="image/png"
                )
            
            st.info("Note: The certificate displayed above is a simulated image. In a full implementation, this would be a downloadable PDF.")

def candidate_login_section():
    """Handles candidate user login."""
    if "candidate_authenticated" not in st.session_state:
        st.session_state.candidate_authenticated = False
    if "candidate_username" not in st.session_state:
        st.session_state.candidate_username = None
    
    # Initialize candidate_view_mode: "login" or "register"
    if "candidate_view_mode" not in st.session_state:
        st.session_state.candidate_view_mode = "login" # Default to login view

    if st.session_state.candidate_authenticated:
        candidate_portal_content()
        return True

    # Use buttons for navigation between login and register
    col_login, col_register = st.columns(2)
    with col_login:
        if st.button("🔐 Candidate Login", key="candidate_login_btn"):
            st.session_state.candidate_view_mode = "login"
            st.rerun()
    with col_register:
        if st.button("📝 Register New Candidate Account", key="candidate_register_btn"):
            st.session_state.candidate_view_mode = "register"
            st.rerun()

    if st.session_state.candidate_view_mode == "login":
        st.subheader("🔐 Candidate Login")
        st.info("If you don't have an account, click 'Register New Candidate Account'.")
        with st.form("candidate_login_form", clear_on_submit=False):
            username = st.text_input("Username", key="candidate_username_login")
            password = st.text_input("Password", type="password", key="candidate_password_login")
            submitted = st.form_submit_button("Login")

            if submitted:
                users = load_users()
                if username not in users:
                    st.error("❌ Invalid username or password. Please register if you don't have an account.")
                else:
                    user_data = users[username]
                    if user_data["status"] == "disabled":
                        st.error("❌ Your account has been disabled. Please contact an administrator.")
                    elif user_data["role"] != "candidate" and username not in ADMIN_USERNAME: # Only candidates or Admin can login here
                        st.error("❌ This login is for Candidate accounts only. Please use the HR Portal if you are an HR user.")
                    elif check_password(password, user_data["password"]):
                        st.session_state.candidate_authenticated = True
                        st.session_state.candidate_username = username
                        st.session_state.candidate_user_company = user_data.get("company", "N/A")
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password.")
    
    elif st.session_state.candidate_view_mode == "register":
        register_section(role="candidate")

    return st.session_state.candidate_authenticated

# Helper function to check if the current HR user is an admin
def is_current_hr_user_admin():
    return st.session_state.get("hr_authenticated", False) and st.session_state.get("hr_username") in ADMIN_USERNAME

# Main application logic
if __name__ == "__main__":
    st.set_page_config(page_title="ScreenerPro", layout="centered")
    st.title("ScreenerPro")
    st.markdown("---")

    # Ensure all admin users exist for testing
    users = load_users()
    default_admin_password = "adminpass" # Define a default password for new admin users

    for admin_user in ADMIN_USERNAME:
        if admin_user not in users:
            users[admin_user] = {"password": hash_password(default_admin_password), "status": "active", "company": "AdminCo", "role": "hr"}
            st.info(f"Created default admin user: {admin_user} with password '{default_admin_password}'")
    save_users(users) # Save after potentially adding new admin users

    # Top-level portal selection - now using buttons
    st.subheader("Select Your Portal:")
    col_hr_portal, col_candidate_portal = st.columns(2)

    with col_hr_portal:
        if st.button("🏢 Go to HR Portal", key="go_to_hr_portal_btn"):
            st.session_state.current_portal = "HR"
            st.session_state.hr_authenticated = False # Reset HR auth state
            st.session_state.candidate_authenticated = False # Ensure candidate is logged out
            st.rerun()
    with col_candidate_portal:
        if st.button("🧑‍💻 Go to Candidate Portal", key="go_to_candidate_portal_btn"):
            st.session_state.current_portal = "Candidate"
            st.session_state.candidate_authenticated = False # Reset Candidate auth state
            st.session_state.hr_authenticated = False # Ensure HR is logged out
            st.rerun()

    # Initialize current_portal if not set (first load)
    if "current_portal" not in st.session_state:
        st.session_state.current_portal = None # No portal selected initially

    # Display content based on selected portal
    if st.session_state.current_portal == "HR":
        if hr_login_section():
            st.markdown("---")
            st.header(f"HR Dashboard - Welcome, {st.session_state.hr_username}!")
            st.write(f"Your company: {st.session_state.get('hr_user_company', 'N/A')}")
            st.write("You are logged in to the HR Portal.")
            
            if is_current_hr_user_admin():
                st.markdown("---")
                st.header("Admin Management Section")
                admin_registration_section()
                admin_password_reset_section()
                admin_disable_enable_user_section()

                st.subheader("👥 All Registered Users (Admin View):")
                try:
                    users_data = load_users()
                    if users_data:
                        display_users = []
                        for user, data in users_data.items():
                            hashed_pass = data.get("password", "N/A")
                            status = data.get("status", "N/A")
                            company = data.get("company", "N/A")
                            role = data.get("role", "N/A")
                            display_users.append([user, hashed_pass, status, company, role])
                        st.dataframe(pd.DataFrame(display_users, columns=["Email/Username", "Hashed Password (DO NOT EXPOSE)", "Status", "Company", "Role"]), use_container_width=True)
                    else:
                        st.info("No users registered yet.")
                except Exception as e:
                    st.error(f"Error loading user data: {e}")
            else:
                st.info(f"Log in as one of the admin accounts ({', '.join(ADMIN_USERNAME)}) to see admin features.")
                
            if st.button("Logout from HR Portal"):
                st.session_state.hr_authenticated = False
                st.session_state.pop('hr_username', None)
                st.session_state.pop('hr_user_company', None)
                st.session_state.current_portal = None # Go back to portal selection
                st.rerun()
        else:
            st.info("Please login or register for the HR Portal.")

    elif st.session_state.current_portal == "Candidate":
        if candidate_login_section():
            st.markdown("---")
            if st.button("Logout from Candidate Portal"):
                st.session_state.candidate_authenticated = False
                st.session_state.pop('candidate_username', None)
                st.session_state.pop('candidate_user_company', None)
                st.session_state.pop('uploaded_resume_name', None)
                st.session_state.pop('screening_score', None)
                st.session_state.pop('screening_feedback', None)
                st.session_state.pop('show_candidate_certificate', None)
                st.session_state.pop('is_screening', None)
                st.session_state.pop('last_simulated_score', None)
                st.session_state.current_portal = None # Go back to portal selection
                st.rerun()
        else:
            st.info("Please login or register for the Candidate Portal.")
    else:
        st.info("Please select a portal to begin.")

