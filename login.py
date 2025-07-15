import streamlit as st
import json
import os
import re
from datetime import datetime
import uuid # For generating unique IDs for mock users
import requests # For making HTTP requests to Firebase Auth and Firestore REST APIs

# Import utility functions and constants from firebase_utils
from firebase_utils import (
    FIREBASE_PROJECT_ID, FIRESTORE_BASE_URL, FIREBASE_WEB_API_KEY,
    get_firestore_document, set_firestore_document, delete_firestore_document, # <--- Changed update_firestore_document to set_firestore_document
    fetch_firestore_collection, log_activity_to_firestore, to_firestore_format
)

# Firebase Auth specific URLs (derived from firebase_utils's FIREBASE_WEB_API_KEY)
FIREBASE_AUTH_SIGNUP_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_WEB_API_KEY}"
FIREBASE_AUTH_SIGNIN_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
FIREBASE_AUTH_RESET_PASSWORD_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={FIREBASE_WEB_API_KEY}"
FIREBASE_AUTH_LOOKUP_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={FIREBASE_WEB_API_KEY}"
FIREBASE_AUTH_UPDATE_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:update?key={FIREBASE_WEB_API_KEY}"


# Admin usernames (these users will have 'recruiter_admin' user_type in their Firestore profile)
ADMIN_USERNAME_EMAILS = ("admin@forscreenerpro", "admin@forscreenerpro2", "manav.nagpal2005@gmail.com")


# --- Firebase Authentication Functions (REST API) ---
def firebase_register_user(email, password):
    """Registers a new user with Firebase Authentication."""
    payload = json.dumps({
        "email": email,
        "password": password,
        "returnSecureToken": True
    })
    try:
        response = requests.post(FIREBASE_AUTH_SIGNUP_URL, data=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Registration failed: {e.response.json().get('error', {}).get('message', 'Unknown error')}")
        return None

def firebase_sign_in_user(email, password):
    """Signs in a user with Firebase Authentication."""
    payload = json.dumps({
        "email": email,
        "password": password,
        "returnSecureToken": True
    })
    try:
        response = requests.post(FIREBASE_AUTH_SIGNIN_URL, data=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Login failed: {e.response.json().get('error', {}).get('message', 'Unknown error')}")
        return None

def firebase_send_password_reset_email(email):
    """Sends a password reset email via Firebase Authentication."""
    payload = json.dumps({
        "requestType": "PASSWORD_RESET",
        "email": email
    })
    try:
        response = requests.post(FIREBASE_AUTH_RESET_PASSWORD_URL, data=payload)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Password reset failed: {e.response.json().get('error', {}).get('message', 'Unknown error')}")
        return False

def firebase_get_user_info(id_token):
    """Looks up user information by ID token."""
    payload = json.dumps({
        "idToken": id_token
    })
    try:
        response = requests.post(FIREBASE_AUTH_LOOKUP_URL, data=payload)
        response.raise_for_status()
        return response.json()['users'][0]
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch user info: {e.response.json().get('error', {}).get('message', 'Unknown error')}")
        return None

def firebase_update_user_profile_auth(id_token, display_name=None, photo_url=None, email=None, password=None):
    """Updates user profile information in Firebase Auth."""
    update_data = {"idToken": id_token}
    if display_name: update_data["displayName"] = display_name
    if photo_url: update_data["photo_url"] = photo_url # Corrected key
    if email: update_data["email"] = email
    if password: update_data["password"] = password
    update_data["returnSecureToken"] = True

    payload = json.dumps(update_data)
    try:
        response = requests.post(FIREBASE_AUTH_UPDATE_URL, data=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to update user profile in Auth: {e.response.json().get('error', {}).get('message', 'Unknown error')}")
        return None


# --- User Profile Management in Firestore (for custom data like company, status, user_type) ---
def get_user_profile_from_firestore(uid):
    """Fetches a user's custom profile from Firestore."""
    profile_path = f"artifacts/{FIREBASE_PROJECT_ID}/users/{uid}/profile"
    return get_firestore_document(profile_path, "user_data")

def set_user_profile_in_firestore(uid, profile_data):
    """Sets/updates a user's custom profile in Firestore."""
    profile_path = f"artifacts/{FIREBASE_PROJECT_ID}/users/{uid}/profile"
    return set_firestore_document(profile_path, "user_data", profile_data) # <--- Changed to set_firestore_document

def get_all_user_profiles_for_admin():
    """
    Fetches all user profiles from Firestore for admin view.
    NOTE: This is a simplified approach for demonstration. In a production app,
    fetching *all* user profiles directly from client-side via REST API is not scalable
    or secure. It's typically done via Firebase Admin SDK on a secure backend.
    """
    st.warning("Admin: Fetching all user profiles directly from Firestore via REST API for management is complex. A production app would use Firebase Admin SDK on a secure backend to list all users.")
    
    # Mocking user list for admin panel demonstration
    mock_users = [
        {"id": "mock_uid_recruiter1", "email": "recruiter1@example.com", "company": "Acme Corp", "status": "active", "user_type": "recruiter_admin"},
        {"id": "mock_uid_candidate1", "email": "candidate1@example.com", "company": "N/A", "status": "active", "user_type": "candidate"},
        {"id": "mock_uid_disabled", "email": "disabled@example.com", "company": "Globex", "status": "disabled", "user_type": "recruiter_admin"},
    ]
    # Add actual logged-in admin to the mock list if not present
    if st.session_state.get('username') and st.session_state.get('username') in ADMIN_USERNAME_EMAILS:
        admin_profile = {
            "id": st.session_state.get('user_id', str(uuid.uuid4())),
            "email": st.session_state.username,
            "company": st.session_state.get('user_company', 'Admin'),
            "status": "active",
            "user_type": "recruiter_admin"
        }
        # Avoid adding duplicates
        if not any(u['email'] == admin_profile['email'] for u in mock_users):
            mock_users.append(admin_profile)

    return mock_users


# --- Email Validation ---
def is_valid_email(email):
    """Validates an email address format."""
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)


# --- Authentication Sections ---
def login_section():
    """
    Handles the login process for both recruiter/admin and candidates using Firebase Auth.
    Returns True if authenticated, False otherwise.
    Sets st.session_state variables upon successful authentication.
    """
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "user_id" not in st.session_state: # Firebase UID
        st.session_state.user_id = None
    if "user_company" not in st.session_state:
        st.session_state.user_company = None
    if "user_type" not in st.session_state: # 'recruiter_admin' or 'candidate'
        st.session_state.user_type = None
    if "id_token" not in st.session_state:
        st.session_state.id_token = None
    if "active_login_tab_selection" not in st.session_state:
        st.session_state.active_login_tab_selection = "Login"
    if "show_forgot_password" not in st.session_state:
        st.session_state.show_forgot_password = False

    if st.session_state.authenticated:
        return True

    st.sidebar.empty() # Clear sidebar content before login form

    login_type = st.radio(
        "Select Login Type:",
        ["Recruiter/Admin Login", "Candidate Login"],
        key="login_type_selector"
    )

    if login_type == "Recruiter/Admin Login":
        st.subheader("🔐 Recruiter/Admin Login")
        username = st.text_input("Username (Email)", key="recruiter_username")
        password = st.text_input("Password", type="password", key="recruiter_password")
        
        col_login_btns = st.columns(2)
        with col_login_btns[0]:
            login_button = st.button("Login with Email/Password")
        with col_login_btns[1]:
            google_login_button = st.button("Login with Google (Mock)")

        if login_button:
            if not username or not password:
                st.error("Please enter both username and password.")
            else:
                with st.spinner("Logging in..."):
                    auth_response = firebase_sign_in_user(username, password)
                    if auth_response:
                        uid = auth_response['localId']
                        id_token = auth_response['idToken']
                        
                        user_profile = get_user_profile_from_firestore(uid)
                        
                        if user_profile and user_profile.get("status") == "disabled":
                            st.error("❌ Your account has been disabled. Please contact an administrator.")
                            log_activity_to_firestore(f"Login attempt for disabled user '{username}'.", user=username)
                        elif user_profile and user_profile.get("user_type") == 'recruiter_admin':
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.session_state.user_id = uid
                            st.session_state.user_company = user_profile.get("company", "N/A")
                            st.session_state.user_type = 'recruiter_admin'
                            st.session_state.id_token = id_token
                            st.success("✅ Login successful!")
                            log_activity_to_firestore(f"User '{username}' logged in via Firebase.", user=username)
                            # Do not call load_session_data_from_firestore_rest here, it's main.py's responsibility
                            st.rerun()
                        else:
                            st.error("❌ Login failed: Invalid credentials or account type. Please ensure you are logging into the correct account type.")
                            log_activity_to_firestore(f"Login failed for '{username}': Invalid credentials or account type.", user=username)
                    # Error message already displayed by firebase_sign_in_user

        if google_login_button:
            st.info("Simulating Google Login...")
            # Generate a mock UID and email for Google login
            mock_uid = f"google_uid_{str(uuid.uuid4())}"
            mock_email = f"google_user_{str(uuid.uuid4())[:8]}@example.com" # Unique mock email
            
            # Check if this mock Google user already has a profile
            user_profile = get_user_profile_from_firestore(mock_uid)
            
            if not user_profile:
                # Create a new profile for the mock Google user
                profile_data = {
                    "email": mock_email,
                    "company": "Mock Google Company",
                    "status": "active",
                    "user_type": "recruiter_admin", # Default mock Google users to recruiter/admin
                    "registered_at": datetime.now(),
                    "login_method": "google_mock"
                }
                success, _ = set_user_profile_in_firestore(mock_uid, profile_data)
                if not success:
                    st.error("Failed to create mock Google user profile in Firestore.")
                    return

            st.session_state.authenticated = True
            st.session_state.username = mock_email
            st.session_state.user_id = mock_uid
            st.session_state.user_company = user_profile.get("company", "Mock Google Company") if user_profile else "Mock Google Company"
            st.session_state.user_type = user_profile.get("user_type", "recruiter_admin") if user_profile else "recruiter_admin"
            st.session_state.id_token = "MOCK_GOOGLE_ID_TOKEN" # Placeholder
            st.success("✅ Mock Google Login successful!")
            log_activity_to_firestore(f"Mock Google user '{mock_email}' logged in.", user=mock_email)
            # Do not call load_session_data_from_firestore_rest here, it's main.py's responsibility
            st.rerun()
            st.warning("Note: This is a **mock** Google login. A real implementation requires client-side JavaScript for OAuth.")

        if st.button("Forgot Password?", key="forgot_password_button"):
            st.session_state.show_forgot_password = True
            st.rerun()

        if st.session_state.get("show_forgot_password"):
            st.markdown("---")
            st.subheader("Reset Password")
            reset_email = st.text_input("Enter your registered email:", key="reset_email_input")
            if st.button("Send Reset Link", key="send_reset_link_button"):
                if reset_email:
                    with st.spinner("Sending password reset link..."):
                        if firebase_send_password_reset_email(reset_email):
                            st.success("A password reset link has been sent to your email address.")
                            log_activity_to_firestore(f"Password reset link sent to '{reset_email}'.", user="System") # Logged by system
                            st.session_state.show_forgot_password = False
                            st.rerun()
                        else:
                            st.error("Failed to send reset link. Please check the email address.")
                else:
                    st.warning("Please enter your email to send a reset link.")

    else: # Candidate Login
        st.subheader("👤 Candidate Login")
        candidate_email = st.text_input("Email", key="candidate_email_login")
        candidate_password = st.text_input("Password", type="password", key="candidate_password_login")
        
        col_cand_login_btns = st.columns(2)
        with col_cand_login_btns[0]:
            candidate_login_button = st.button("Login as Candidate")
        with col_cand_login_btns[1]:
            candidate_google_login_button = st.button("Candidate Login with Google (Mock)")

        if candidate_login_button:
            if not candidate_email or not candidate_password:
                st.error("Please enter both email and password.")
            else:
                with st.spinner("Logging in..."):
                    auth_response = firebase_sign_in_user(candidate_email, candidate_password)
                    if auth_response:
                        uid = auth_response['localId']
                        id_token = auth_response['idToken']
                        
                        user_profile = get_user_profile_from_firestore(uid)
                        
                        if user_profile and user_profile.get("status") == "disabled":
                            st.error("❌ Your account has been disabled. Please contact support.")
                            log_activity_to_firestore(f"Login attempt for disabled candidate '{candidate_email}'.", user=candidate_email)
                        elif user_profile and user_profile.get("user_type") == 'candidate':
                            st.session_state.authenticated = True
                            st.session_state.username = candidate_email
                            st.session_state.user_id = uid
                            st.session_state.user_company = user_profile.get("company", "N/A") # Candidates might not have a company
                            st.session_state.user_type = 'candidate'
                            st.session_state.id_token = id_token
                            st.success(f"✅ Welcome, {candidate_email}!")
                            log_activity_to_firestore(f"Candidate '{candidate_email}' logged in via Firebase.", user=candidate_email)
                            # Do not call load_session_data_from_firestore_rest here, it's main.py's responsibility
                            st.rerun()
                        else:
                            st.error("❌ Login failed: Invalid credentials or account type. Please ensure you are logging into the correct account type.")
                            log_activity_to_firestore(f"Login failed for candidate '{candidate_email}': Invalid credentials or account type.", user=candidate_email)
                    # Error message already displayed by firebase_sign_in_user

        if candidate_google_login_button:
            st.info("Simulating Candidate Google Login...")
            mock_uid = f"google_candidate_uid_{str(uuid.uuid4())}"
            mock_email = f"google_candidate_{str(uuid.uuid4())[:8]}@example.com"
            
            user_profile = get_user_profile_from_firestore(mock_uid)
            
            if not user_profile:
                profile_data = {
                    "email": mock_email,
                    "company": "N/A", # Candidates typically don't have a company in this context
                    "status": "active",
                    "user_type": "candidate",
                    "registered_at": datetime.now(),
                    "login_method": "google_mock"
                }
                success, _ = set_user_profile_in_firestore(mock_uid, profile_data)
                if not success:
                    st.error("Failed to create mock Google candidate profile in Firestore.")
                    return

            st.session_state.authenticated = True
            st.session_state.username = mock_email
            st.session_state.user_id = mock_uid
            st.session_state.user_company = user_profile.get("company", "N/A") if user_profile else "N/A"
            st.session_state.user_type = user_profile.get("user_type", "candidate") if user_profile else "candidate"
            st.session_state.id_token = "MOCK_GOOGLE_ID_TOKEN"
            st.success("✅ Mock Google Candidate Login successful!")
            log_activity_to_firestore(f"Mock Google candidate '{mock_email}' logged in.", user=mock_email)
            # Do not call load_session_data_from_firestore_rest here, it's main.py's responsibility
            st.rerun()
            st.warning("Note: This is a **mock** Google login. A real implementation requires client-side JavaScript for OAuth.")

        st.markdown("---")
        st.info("Don't have a candidate account? Register below.")
        
        st.subheader("📝 Candidate Registration")
        with st.form("candidate_registration_form", clear_on_submit=True):
            new_candidate_email = st.text_input("Candidate Email", key="new_candidate_email_reg")
            new_candidate_password = st.text_input("Password", type="password", key="new_candidate_password_reg")
            confirm_candidate_password = st.text_input("Confirm Password", type="password", key="confirm_candidate_password_reg")
            
            submitted_cand_reg = st.form_submit_button("Register Candidate Account")

            if submitted_cand_reg:
                if not new_candidate_email or not new_candidate_password or not confirm_candidate_password:
                    st.error("All fields are required.")
                elif not is_valid_email(new_candidate_email):
                    st.error("Please enter a valid email address.")
                elif new_candidate_password != confirm_candidate_password:
                    st.error("Passwords do not match.")
                else:
                    with st.spinner("Registering candidate..."):
                        auth_response = firebase_register_user(new_candidate_email, new_candidate_password)
                        if auth_response:
                            uid = auth_response['localId']
                            
                            profile_data = {
                                "email": new_candidate_email,
                                "company": "N/A", # Candidates typically don't have a company
                                "status": "active",
                                "user_type": "candidate", # Mark as candidate
                                "registered_at": datetime.now()
                            }
                            success, _ = set_user_profile_in_firestore(uid, profile_data)
                            
                            if success:
                                st.success("✅ Candidate account registered successfully! You can now log in.")
                                log_activity_to_firestore(f"New candidate '{new_candidate_email}' registered via Firebase.", user=new_candidate_email)
                                st.session_state.active_login_tab_selection = "Login"
                                st.rerun()
                            else:
                                st.error("Registration successful, but failed to save candidate profile. Please contact support.")
                                log_activity_to_firestore(f"Candidate '{new_candidate_email}' registered, but Firestore profile save failed.", user=new_candidate_email)
                        # Error message already displayed by firebase_register_user

    return st.session_state.authenticated

# Helper function to check if the current user is an admin
def is_current_user_admin():
    # Admin check relies on user_type in session state, set from Firestore profile
    return st.session_state.get("authenticated", False) and st.session_state.get("user_type") == 'recruiter_admin' and st.session_state.get("username") in ADMIN_USERNAME_EMAILS

def is_current_user_candidate():
    return st.session_state.get("authenticated", False) and st.session_state.get("user_type") == 'candidate'

# Admin-driven user creation form using Firebase Auth.
def admin_registration_section():
    st.subheader("➕ Create New User Account (Admin Only)")
    with st.form("admin_registration_form", clear_on_submit=True):
        new_username = st.text_input("New User's Username (Email)", key="new_username_admin_reg")
        new_company_name = st.text_input("New User's Company Name", key="new_company_name_admin_reg")
        new_password = st.text_input("New User's Password", type="password", key="new_password_admin_reg")
        new_user_type = st.selectbox("User Type:", ["recruiter_admin", "candidate"], key="new_user_type_admin_reg")
        admin_register_button = st.form_submit_button("Add New User")

    if admin_register_button:
        if not new_username or not new_password or not new_company_name:
            st.error("Please fill in all fields.")
        elif not is_valid_email(new_username):
            st.error("Please enter a valid email address for the username.")
        else:
            with st.spinner(f"Adding user '{new_username}'..."):
                auth_response = firebase_register_user(new_username, new_password)
                if auth_response:
                    uid = auth_response['localId']
                    
                    profile_data = {
                        "email": new_username,
                        "company": new_company_name,
                        "status": "active",
                        "user_type": new_user_type,
                        "registered_at": datetime.now(),
                        "added_by_admin": st.session_state.get('username', 'Admin')
                    }
                    success, _ = set_user_profile_in_firestore(uid, profile_data)
                    
                    if success:
                        st.success(f"✅ User '{new_username}' added successfully!")
                        log_activity_to_firestore(f"Admin '{st.session_state.username}' added new user '{new_username}' as '{new_user_type}'.", user=st.session_state.username)
                    else:
                        st.error(f"User '{new_username}' created, but failed to save profile to Firestore.")
                        log_activity_to_firestore(f"Admin '{st.session_state.username}' added user '{new_username}', but Firestore profile save failed.", user=st.session_state.username)
                else:
                    st.error(f"Failed to add user '{new_username}' via Firebase Auth.")

# Admin-driven password reset form using Firebase Auth.
def admin_password_reset_section():
    st.subheader("🔑 Reset User Password (Admin Only)")
    
    all_users = get_all_user_profiles_for_admin()
    user_options = [user['email'] for user in all_users if user['email'] not in ["admin@forscreenerpro", "admin@forscreenerpro2", "manav.nagpal2005@gmail.com"]] # Explicitly filter out admin emails

    if not user_options:
        st.info("No non-admin users available to reset passwords for.")
        selected_user_email = None
    else:
        selected_user_email = st.selectbox("Select User to Reset Password For", user_options, key="reset_user_select")

    with st.form("admin_reset_password_form", clear_on_submit=True):
        reset_button = st.form_submit_button("Send Password Reset Email")

        if reset_button:
            if not selected_user_email:
                st.warning("Please select a user to reset their password.")
            else:
                with st.spinner(f"Sending password reset email to {selected_user_email}..."):
                    if firebase_send_password_reset_email(selected_user_email):
                        st.success(f"✅ Password reset email sent to '{selected_user_email}'.")
                        log_activity_to_firestore(f"Admin '{st.session_state.username}' sent password reset email to '{selected_user_email}'.", user=st.session_state.username)
                    else:
                        st.error(f"Failed to send password reset email to '{selected_user_email}'.")


# Admin-driven user disable/enable form and user list using Firebase Auth and Firestore.
def admin_disable_enable_user_section():
    st.subheader("⛔ Toggle User Status (Admin Only)")
    
    all_users = get_all_user_profiles_for_admin()
    non_admin_users = [user for user in all_users if user['email'] not in ["admin@forscreenerpro", "admin@forscreenerpro2", "manav.nagpal2005@gmail.com"]] # Explicitly filter out admin emails

    user_data_for_table = []
    for user in non_admin_users:
        user_data_for_table.append({
            "Username (Email)": user['email'],
            "Company Name": user.get("company", "N/A"),
            "User Type": user.get("user_type", "N/A").replace('_', ' ').title(),
            "Status": user.get("status", "N/A").capitalize()
        })
    
    st.markdown("### Registered Users Overview")
    if user_data_for_table:
        import pandas as pd # Import pandas here as it's used only in this function
        st.dataframe(pd.DataFrame(user_data_for_table), use_container_width=True, hide_index=True)
    else:
        st.info("No non-admin users registered yet.")
    st.markdown("---")

    user_options = [user['email'] for user in non_admin_users]

    if not user_options:
        st.info("No non-admin users available to manage status for.")
        selected_user_email = None
    else:
        selected_user_email = st.selectbox("Select User to Toggle Status", user_options, key="toggle_user_select")

    if selected_user_email:
        selected_user_profile = next((u for u in non_admin_users if u['email'] == selected_user_email), None)
        if selected_user_profile:
            current_status = selected_user_profile.get("status", "active")
            st.info(f"Current status of '{selected_user_email}': **{current_status.upper()}**")

            if st.button(f"Toggle to {'Disable' if current_status == 'active' else 'Enable'} User", key="toggle_status_button"):
                new_status = "disabled" if current_status == "active" else "active"
                uid_to_update = selected_user_profile['id'] # Use the 'id' from the mock user list
                
                with st.spinner(f"Updating status for {selected_user_email}..."):
                    success, _ = set_firestore_document(uid_to_update, {"status": new_status, "last_updated": datetime.now()}) # <--- Changed to set_firestore_document
                    if success:
                        st.success(f"✅ User '{selected_user_email}' status set to **{new_status.upper()}**.")
                        log_activity_to_firestore(f"Admin '{st.session_state.username}' toggled status of '{selected_user_email}' to '{new_status}'.", user=st.session_state.username)
                        st.rerun()
                    else:
                        st.error(f"Failed to update status for '{selected_user_email}'.")
        else:
            st.warning("Selected user profile not found. Please refresh the list.")
