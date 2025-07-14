# email_sender.py (or email_page.py - make sure import in main.py matches)

import streamlit as st
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import os

def send_email_to_candidate():
    st.markdown("## 📤 Email Candidates")
    st.info("Prepare and send emails to shortlisted candidates based on screening results.")

    # Check if screening results are available in session state
    if 'screening_results' not in st.session_state or not st.session_state['screening_results']:
        st.warning("No screening results found. Please run the '🧠 Resume Screener' first to get candidates to email.")
        return # Exit the function if no results

    try:
        # It's safer to create a DataFrame from the list of dicts stored in session state
        # directly within the function where it's used, as session_state might hold a list.
        df_results = pd.DataFrame(st.session_state['screening_results'])

        # Ensure required columns exist before proceeding
        required_columns = ['Candidate Name', 'Email', 'Score (%)', 'Years Experience', 'AI Suggestion']
        missing_columns = [col for col in required_columns if col not in df_results.columns]

        if missing_columns:
            st.error(f"Missing essential data columns in screening results: {', '.join(missing_columns)}. "
                     "Please ensure the 'Resume Screener' generated these columns.")
            # Display available columns for debugging
            st.write(f"Available columns: {list(df_results.columns)}")
            return

        # Filtering for shortlisted candidates based on criteria (can be adjusted)
        # It's better to get the cutoff values from session_state if they are stored there by screener.py
        cutoff_score = st.session_state.get('screening_cutoff_score', 75)
        min_exp_required = st.session_state.get('screening_min_experience', 2)

        shortlisted_candidates = df_results[
            (df_results["Score (%)"] >= cutoff_score) &
            (df_results["Years Experience"] >= min_exp_required)
        ].copy() # Use .copy() to avoid SettingWithCopyWarning

        if shortlisted_candidates.empty:
            st.warning(f"No candidates meet the current shortlisting criteria (Score >= {cutoff_score}%, Experience >= {min_exp_required} years). Adjust criteria in Screener or review results.")
            st.dataframe(df_results[['Candidate Name', 'Email', 'Score (%)', 'Years Experience', 'AI Suggestion']], use_container_width=True)
            return

        st.success(f"Found {len(shortlisted_candidates)} shortlisted candidates.")
        st.dataframe(shortlisted_candidates[['Candidate Name', 'Email', 'Score (%)', 'AI Suggestion']], use_container_width=True)

        st.markdown("### 📧 Email Configuration")
        sender_email = st.text_input("Your Email (Sender)", key="sender_email")
        sender_password = st.text_input("Your Email Password (App Password)", type="password", key="sender_password")
        smtp_server = st.text_input("SMTP Server", "smtp.gmail.com", key="smtp_server")
        smtp_port = st.number_input("SMTP Port", 587, key="smtp_port")

        st.markdown("### ✍️ Email Content")
        email_subject = st.text_input("Email Subject", "Job Application Update - Your Application to [Job Title]")
        default_body = """
        Dear {candidate_name},

        Thank you for your application for the position of [Job Title] at [Company Name].

        We have reviewed your resume and would like to provide an update. Based on our initial assessment, your profile showed a score of {score_percent:.1f}% and {years_experience:.1f} years of experience.

        Our AI's suggestion for your profile: {ai_suggestion}

        We will be in touch shortly regarding the next steps in our hiring process.

        Best regards,

        The [Company Name] Hiring Team
        """
        email_body = st.text_area("Email Body (use {candidate_name}, {score_percent}, {years_experience}, {ai_suggestion})", default_body, height=300)

        if st.button("🚀 Send Emails to Shortlisted Candidates"):
            if not sender_email or not sender_password:
                st.error("Please enter your sender email and password.")
                return

            try:
                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls() # Secure the connection
                    server.login(sender_email, sender_password)

                    for index, row in shortlisted_candidates.iterrows():
                        candidate_name = row['Candidate Name']
                        candidate_email = row['Email']
                        score_percent = row['Score (%)']
                        years_experience = row['Years Experience']
                        ai_suggestion = row['AI Suggestion']

                        # Format the email body with actual candidate data
                        formatted_body = email_body.format(
                            candidate_name=candidate_name,
                            score_percent=score_percent,
                            years_experience=years_experience,
                            ai_suggestion=ai_suggestion
                        )

                        msg = MIMEMultipart()
                        msg['From'] = sender_email
                        msg['To'] = candidate_email
                        msg['Subject'] = email_subject
                        msg.attach(MIMEText(formatted_body, 'plain'))

                        server.send_message(msg)
                        st.success(f"Email sent to {candidate_name} ({candidate_email})")
                        
                        # Add sent email to session state for tracking
                        if 'sent_emails_log' not in st.session_state:
                            st.session_state['sent_emails_log'] = []
                        st.session_state['sent_emails_log'].append({
                            "timestamp": pd.Timestamp.now().isoformat(),
                            "candidate_name": candidate_name,
                            "candidate_email": candidate_email,
                            "subject": email_subject,
                            "body_snippet": formatted_body[:100] + "..." # Log a snippet
                        })

                st.success("All emails sent successfully!")
            except smtplib.SMTPAuthenticationError:
                st.error("Email sending failed: Invalid email or app password. For Gmail, you might need to use an App Password.")
                st.info("If using Gmail, please ensure you've enabled '2-Step Verification' and generated an 'App password' for your app, then use that password here.")
            except smtplib.SMTPConnectError:
                st.error(f"Email sending failed: Could not connect to SMTP server {smtp_server}:{smtp_port}. Check server address and port.")
            except Exception as e:
                st.error(f"An unexpected error occurred during email sending: {e}")

    except Exception as e:
        st.error(f"An error occurred while preparing candidate data: {e}")


# This ensures the function is called when email_sender.py is executed (via exec() or direct import)
if __name__ == "__main__":
    send_email_to_candidate()
