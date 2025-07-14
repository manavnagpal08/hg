# feedback.py
import streamlit as st
import requests

# --- Logging Function ---
def log_user_action(user_email, action, details=None):
    if details:
        print(f"LOG: User '{user_email}' performed action '{action}' with details: {details}")
    else:
        print(f"LOG: User '{user_email}' performed action '{action}'")

# --- Feedback Page Function ---
def feedback_and_help_page():
    user_email = st.session_state.get('user_email', 'anonymous')
    log_user_action(user_email, "FEEDBACK_HELP_PAGE_ACCESSED")

    st.markdown("""
    <style>
    .screener-container {
        background-color: #f9f9ff;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="screener-container">', unsafe_allow_html=True)
    st.markdown("## ❓ Feedback")
    st.caption("We value your input! Please use the form below to send us your feedback or questions.")

    with st.form("feedback_form", clear_on_submit=True):
        feedback_name = st.text_input("Your Name (Optional)", key="feedback_name")
        feedback_email = st.text_input("Your Email (Optional)", key="feedback_email")
        feedback_subject = st.text_input("Subject", "Feedback on ScreenerPro", key="feedback_subject")
        feedback_message = st.text_area("Your Message", height=150, key="feedback_message")
        
        submit_button = st.form_submit_button("Send Feedback")

        if submit_button:
            if not feedback_message.strip():
                st.error("❌ Please enter your message before sending feedback.")
                log_user_action(user_email, "FEEDBACK_SUBMIT_FAILED", {"reason": "Empty message"})
            else:
                # ✅ Send to Formspree
                formspree_url = "https://formspree.io/f/mwpqevno"  # Your endpoint
                payload = {
                    "name": feedback_name,
                    "email": feedback_email,
                    "subject": feedback_subject,
                    "message": feedback_message
                }

                response = requests.post(formspree_url, data=payload)

                if response.status_code == 200:
                    st.success("✅ Thank you! Your feedback has been submitted successfully.")
                    log_user_action(user_email, "FEEDBACK_SUBMITTED_FORMSPREE", {"subject": feedback_subject})
                else:
                    st.error("⚠️ Something went wrong. Please try again later.")
                    log_user_action(user_email, "FEEDBACK_SUBMIT_FAILED", {"status": response.status_code})

    st.markdown("</div>", unsafe_allow_html=True)
