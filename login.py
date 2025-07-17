import streamlit as st
import json
import bcrypt
import os
import re

USER_DB_FILE = "users.json"
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
            elif "status" not in data:
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

def register_form(role_label):
    st.subheader(f"📝 Register as {role_label}")
    with st.form(f"{role_label}_register_form", clear_on_submit=True):
        new_username = st.text_input("Email", key=f"{role_label}_new_user")
        new_company = st.text_input("Company", key=f"{role_label}_new_company")
        new_password = st.text_input("Password", type="password", key=f"{role_label}_new_pass")
        confirm_password = st.text_input("Confirm Password", type="password", key=f"{role_label}_confirm_pass")
        submit = st.form_submit_button("Register")

        if submit:
            if not new_username or not new_company or not new_password or not confirm_password:
                st.error("Please fill in all fields.")
            elif not is_valid_email(new_username):
                st.error("Enter a valid email.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                users = load_users()
                if new_username in users:
                    st.error("User already exists.")
                else:
                    users[new_username] = {
                        "password": hash_password(new_password),
                        "status": "active",
                        "company": new_company,
                        "role": "Candidate" if role_label == "Candidate" else "HR"
                    }
                    save_users(users)
                    st.success("✅ Registered successfully! Please login.")

def login_form(role_label):
    st.subheader(f"🔐 Login as {role_label}")
    with st.form(f"{role_label}_login_form", clear_on_submit=False):
        username = st.text_input("Email", key=f"{role_label}_login_user")
        password = st.text_input("Password", type="password", key=f"{role_label}_login_pass")
        submit = st.form_submit_button("Login")

        if submit:
            users = load_users()
            if username not in users:
                st.error("Invalid username or password.")
            else:
                user = users[username]
                if user["status"] == "disabled":
                    st.error("Your account is disabled.")
                elif user["role"] != role_label:
                    st.error(f"This account is not a {role_label} account.")
                elif not check_password(password, user["password"]):
                    st.error("Invalid credentials.")
                else:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.user_company = user.get("company", "N/A")
                    st.session_state.user_role = user.get("role", "HR")
                    st.success(f"✅ {role_label} login successful!")
                    st.rerun()

def login_section():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    hr_tab, candidate_tab = st.tabs(["HR Portal", "Candidate Portal"])

    with hr_tab:
        option = st.radio("Select Action", ["Login", "Register"], key="hr_action")
        if option == "Login":
            login_form("HR")
        else:
            register_form("HR")

    with candidate_tab:
        option = st.radio("Select Action", ["Login", "Register"], key="cand_action")
        if option == "Login":
            login_form("Candidate")
        else:
            register_form("Candidate")

    return st.session_state.authenticated

def is_current_user_admin():
    return st.session_state.get("username") in ADMIN_USERNAME

# If running directly
if __name__ == "__main__":
    st.set_page_config("ScreenerPro Login", layout="centered")
    st.title("🔐 ScreenerPro Login System")

    # Create default users if needed
    users = load_users()
    if "candidate@demo.com" not in users:
        users["candidate@demo.com"] = {
            "password": hash_password("candpass"),
            "status": "active",
            "company": "DemoCo",
            "role": "Candidate"
        }
    if "hr@demo.com" not in users:
        users["hr@demo.com"] = {
            "password": hash_password("hrpass"),
            "status": "active",
            "company": "HRCo",
            "role": "HR"
        }
    for admin_user in ADMIN_USERNAME:
        if admin_user not in users:
            users[admin_user] = {
                "password": hash_password("adminpass"),
                "status": "active",
                "company": "AdminCo",
                "role": "HR"
            }
    save_users(users)

    if login_section():
        st.sidebar.success(f"Welcome {st.session_state.username} ({st.session_state.user_role})")
        if st.session_state.user_role == "Candidate":
            st.header("🎯 Resume Screener Page")
            st.write("Only visible to Candidates.")
            # Insert resume screener here
        else:
            st.header("🏢 Full HR Dashboard")
            st.write("Visible to HR/Admin only.")
            # Insert full app/dashboard here

        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.pop("username", None)
            st.session_state.pop("user_role", None)
            st.session_state.pop("user_company", None)
            st.rerun()
