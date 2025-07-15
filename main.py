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
import requests

# --- Firebase REST Setup ---
FIREBASE_WEB_API_KEY = "AIzaSyDkYourRealAPIKey12345"  # Replace with your real Web API Key
FIREBASE_PROJECT_ID = "screenerproapp"  # Replace with your Firebase project ID
FIREBASE_FIRESTORE_URL = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents"

# --- Save session data to Firestore ---
def save_session_data_to_firestore_rest(username, session_data):
    try:
        if not username:
            st.warning("No username found. Please log in.")
            return

        doc_path = f"artifacts/{FIREBASE_PROJECT_ID}/users/{username}/session_data/current_session"
        url = f"{FIREBASE_FIRESTORE_URL}/{doc_path}?key={FIREBASE_WEB_API_KEY}"

        data = {
            "fields": {
                key: {"stringValue": str(value)} for key, value in session_data.items()
            }
        }

        res = requests.patch(url, json=data)
        if res.status_code in [200, 201]:
            st.success("✅ Session data saved using REST API.")
        else:
            st.error(f"❌ REST Save failed: {res.status_code}, {res.text}")
    except Exception as e:
        st.error(f"🔥 REST Firebase error: {e}")

# File to store user credentials
USER_DB_FILE = "users.json"
ADMIN_USERNAME = ("admin@forscreenerpro", "admin@forscreenerpro2")

# Load users from file
def load_users():
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "r") as f:
            return json.load(f)
    return {}

# Save users to file
def save_users(users):
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f, indent=4)

# Register user
def register_user(username, password):
    users = load_users()
    if username in users:
        return False
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    users[username] = hashed_pw.decode()
    save_users(users)
    return True

# Authenticate user
def authenticate_user(username, password):
    users = load_users()
    if username in users and bcrypt.checkpw(password.encode(), users[username].encode()):
        return True
    return False

# Login and register UI
def show_login():
    st.title("🔐 Resume Screener Pro - Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    action = st.radio("Action", ["Login", "Register"])
    if st.button("Submit"):
        if action == "Login":
            if authenticate_user(username, password):
                st.session_state.username = username
                st.success(f"Welcome back, {username}!")
            else:
                st.error("Invalid credentials")
        elif action == "Register":
            if register_user(username, password):
                st.success("User registered successfully")
            else:
                st.warning("Username already exists")

# Main dashboard
def show_dashboard():
    st.title("📊 Resume Screening Dashboard")
    st.write(f"Logged in as: `{st.session_state.username}`")

    uploaded_files = st.file_uploader("Upload Resumes (PDFs)", accept_multiple_files=True)
    if uploaded_files:
        st.info("🔍 Resume parsing is under development.")
        st.session_state["comprehensive_df"] = pd.DataFrame({
            "Name": [f"Candidate {i+1}" for i in range(len(uploaded_files))],
            "Score": [round(50 + i * 10, 2) for i in range(len(uploaded_files))]
        })
        st.dataframe(st.session_state["comprehensive_df"])

    # Save to Firebase button
    if st.button("💾 Save to Firebase via REST"):
        if 'comprehensive_df' in st.session_state and not st.session_state['comprehensive_df'].empty:
            save_session_data_to_firestore_rest(
                st.session_state.get('username', 'anonymous'),
                {
                    "timestamp": str(datetime.now()),
                    "screened_count": len(st.session_state['comprehensive_df']),
                    "status": "saved from Streamlit Cloud"
                }
            )
        else:
            st.warning("Nothing to save — please run screening first.")

# Streamlit app flow
if 'username' not in st.session_state:
    show_login()
else:
    show_dashboard()
