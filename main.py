import streamlit as st

# Authentication check
if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.warning("You must login first.")
    st.stop()

# Role-based access control
user_role = st.session_state.get("user_role", "HR")  # default to HR
username = st.session_state.get("username", "unknown")

st.sidebar.title("ScreenerPro Navigation")

# HR/Admin full access
if user_role == "HR":
    page = st.sidebar.radio("Go to", [
        "Home",
        "Resume Screener",
        "Shortlisted Candidates",
        "Performance Analytics",
        "Candidate Reports",
        "Settings"
    ])
else:
    # Candidate access limited to Resume Screener
    page = "Resume Screener"

# Display based on selected page
if page == "Home":
    st.title("🏠 Welcome to ScreenerPro Dashboard")
    st.write(f"Hello **{username}**! You are logged in as **{user_role}**.")
    st.success("You have access to all features.")
elif page == "Resume Screener":
    st.title("📄 Resume Screener")
    st.write(f"Welcome **{username}** ({user_role})")
    st.info("Upload resumes and get smart screening.")
elif page == "Shortlisted Candidates":
    st.title("✅ Shortlisted Candidates")
    st.write("View candidates shortlisted by the screener.")
elif page == "Performance Analytics":
    st.title("📊 Performance Analytics")
    st.write("Visualize screening performance.")
elif page == "Candidate Reports":
    st.title("🧾 Candidate Reports")
    st.write("Download and manage reports.")
elif page == "Settings":
    st.title("⚙️ Settings")
    st.write("Manage app settings and preferences.")
else:
    st.error("Unknown page.")

# Logout option
if st.sidebar.button("Logout"):
    for key in ["authenticated", "username", "user_role", "user_company"]:
        st.session_state.pop(key, None)
    st.rerun()
