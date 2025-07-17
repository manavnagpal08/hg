import streamlit as st
import json
import bcrypt
import os
import re
import pandas as pd

# File to store user credentials
USER_DB_FILE = "users.json"

# Admin usernames
ADMIN_USERNAME = ("admin@forscreenerpro", "admin@forscreenerpro2", "manav.nagpal2005@gmail.com")

def load_users():
    if not os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "w") as f:
            json.dump({}, f)
    with open(USER_DB_FILE, "r") as f:
        users = json.load(f)
        for username, data in users.items():
            if isinstance(data, str):
                users[username] = {"password": data, "status": "active", "company": "N/A", "role": "HR"}
            if "status" not in data:
                data["status"] = "active"
            if "company" not in data:
                data["company"] = "N/A"
            if "role" not in data:
                data["role"] = "HR"
        return users

def save_users(users):
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f, indent=4)

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def register_section():
    st.subheader("📝 Create New Account")
    with st.form("registration_form", clear_on_submit=True):
        new_username = st.text_input("Choose Username (Email address required)")
        new_company_name = st.text_input("Company Name")
        new_password = st.text_input("Choose Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        role = st.selectbox("Registering As", ["HR", "Candidate"])
        register_button = st.form_submit_button("Register New Account")

        if register_button:
            if not new_username or not new_password or not confirm_password or not new_company_name:
                st.error("Please fill in all fields.")
            elif not is_valid_email(new_username):
                st.error("Please enter a valid email address.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                users = load_users()
                if new_username in users:
                    st.error("Username already exists.")
                else:
                    users[new_username] = {
                        "password": hash_password(new_password),
                        "status": "active",
                        "company": new_company_name,
                        "role": role
                    }
                    save_users(users)
                    st.success("✅ Registration successful! Please login now.")
                    st.session_state.active_login_tab_selection = "Login"

def admin_registration_section():
    st.subheader("➕ Create New User Account (Admin Only)")
    with st.form("admin_registration_form", clear_on_submit=True):
        new_username = st.text_input("New User's Username (Email)")
        new_company_name = st.text_input("New User's Company Name")
        new_password = st.text_input("New User's Password", type="password")
        role = st.selectbox("Assign Role", ["HR", "Candidate"])
        admin_register_button = st.form_submit_button("Add New User")

    if admin_register_button:
        if not new_username or not new_password or not new_company_name:
            st.error("Please fill in all fields.")
        elif not is_valid_email(new_username):
            st.error("Please enter a valid email address.")
        else:
            users = load_users()
            if new_username in users:
                st.error(f"User '{new_username}' already exists.")
            else:
                users[new_username] = {
                    "password": hash_password(new_password),
                    "status": "active",
                    "company": new_company_name,
                    "role": role
                }
                save_users(users)
                st.success(f"✅ User '{new_username}' added successfully!")

def admin_password_reset_section():
    st.subheader("🔑 Reset User Password (Admin Only)")
    users = load_users()
    user_options = [user for user in users if user not in ADMIN_USERNAME]

    if not user_options:
        st.info("No users available.")
        return

    with st.form("admin_reset_password_form", clear_on_submit=True):
        selected_user = st.selectbox("Select User", user_options)
        new_password = st.text_input("New Password", type="password")
        reset_button = st.form_submit_button("Reset Password")

        if reset_button:
            if not new_password:
                st.error("Enter a password.")
            else:
                users[selected_user]["password"] = hash_password(new_password)
                save_users(users)
                st.success(f"✅ Password for '{selected_user}' reset.")

def admin_disable_enable_user_section():
    st.subheader("⛔ Toggle User Status (Admin Only)")
    users = load_users()
    user_options = [user for user in users if user not in ADMIN_USERNAME]

    if not user_options:
        st.info("No users to toggle.")
        return

    with st.form("admin_toggle_user_status_form", clear_on_submit=False):
        selected_user = st.selectbox("Select User", user_options)
        current_status = users[selected_user]["status"]
        st.info(f"Current status: **{current_status.upper()}**")

        if st.form_submit_button(f"Toggle to {'Disable' if current_status == 'active' else 'Enable'}"):
            new_status = "disabled" if current_status == "active" else "active"
            users[selected_user]["status"] = new_status
            save_users(users)
            st.success(f"✅ User status updated to {new_status.upper()}.")
            st.rerun()

def login_section():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "role" not in st.session_state:
        st.session_state.role = None

    if "active_login_tab_selection" not in st.session_state:
        if not os.path.exists(USER_DB_FILE) or len(load_users()) == 0:
            st.session_state.active_login_tab_selection = "Register"
        else:
            st.session_state.active_login_tab_selection = "Login"

    if st.session_state.authenticated:
        return True

    tab_selection = st.radio("Select Option:", ("Login", "Register"),
                             index=0 if st.session_state.active_login_tab_selection == "Login" else 1)

    if tab_selection == "Login":
        st.subheader("🔐 Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

            if submitted:
                users = load_users()
                if username not in users:
                    st.error("❌ Invalid credentials.")
                else:
                    user_data = users[username]
                    if user_data["status"] == "disabled":
                        st.error("❌ Your account is disabled.")
                    elif not check_password(password, user_data["password"]):
                        st.error("❌ Incorrect password.")
                    else:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.role = user_data.get("role", "HR")
                        st.session_state.user_company = user_data.get("company", "N/A")
                        st.success(f"✅ Login successful as {st.session_state.role}!")
                        st.rerun()

    elif tab_selection == "Register":
        register_section()

    return st.session_state.authenticated

def is_current_user_admin():
    return st.session_state.get("authenticated", False) and st.session_state.get("username") in ADMIN_USERNAME

# Entry point
if __name__ == "__main__":
    st.set_page_config(page_title="Login/Register", layout="centered")
    st.title("ScreenerPro Authentication")

    users = load_users()
    default_admin_password = "adminpass"

    for admin_user in ADMIN_USERNAME:
        if admin_user not in users:
            users[admin_user] = {
                "password": hash_password(default_admin_password),
                "status": "active",
                "company": "AdminCo",
                "role": "HR"
            }
            st.info(f"Admin user created: {admin_user}")
    save_users(users)

    if login_section():
        st.write(f"Welcome, **{st.session_state.username}**")
        st.write(f"Company: {st.session_state.get('user_company', 'N/A')}")
        st.write(f"Role: **{st.session_state.role}**")

        if st.session_state.role == "Candidate":
            st.success("✅ You can now access the Resume Screener page.")
            st.markdown("👉 [Go to Resume Screener](#)")  # Replace with real link
        elif st.session_state.role == "HR":
            st.success("✅ HR Access Granted.")
            st.markdown("👉 [Go to Dashboard](#)")
            st.markdown("👉 [Go to Resume Screener](#)")

        if is_current_user_admin():
            st.markdown("---")
            st.header("🔐 Admin Panel")
            admin_registration_section()
            admin_password_reset_section()
            admin_disable_enable_user_section()

            st.subheader("👥 All Registered Users")
            try:
                data = []
                for user, info in load_users().items():
                    data.append([user, info["status"], info["company"], info.get("role", "HR")])
                st.dataframe(pd.DataFrame(data, columns=["Email", "Status", "Company", "Role"]), use_container_width=True)
            except Exception as e:
                st.error(f"Error loading users: {e}")

        if st.button("Logout"):
            for key in ["authenticated", "username", "role", "user_company"]:
                st.session_state.pop(key, None)
            st.rerun()
