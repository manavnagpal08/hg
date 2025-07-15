import streamlit as st
import pandas as pd
import os
import datetime

# Import Firebase modules
from firebase_admin import credentials, initialize_app
from firebase_admin import firestore
import json

# Initialize Firebase (if not already initialized)
# This block should ideally be in main.py or a central config file,
# but for self-contained Canvas updates, we'll include it here.
# It checks if __firebase_config and __app_id are defined by the environment.
try:
    if 'firebase_initialized' not in st.session_state:
        # Use the global variables provided by the Canvas environment
        app_id = typeof __app_id !== 'undefined' ? __app_id : 'default-app-id';
        firebase_config = JSON.parse(typeof __firebase_config !== 'undefined' ? __firebase_config : '{}');

        # Initialize Firebase Admin SDK for server-side operations (like Firestore)
        # This part assumes you have a service account JSON file.
        # For Canvas, we'll use a placeholder and rely on the environment's setup.
        # In a real deployment, you'd load your service account credentials.
        
        # Check if an app is already initialized to prevent re-initialization errors
        if not initialize_app(): # No specific app name, checks default app
            cred = credentials.Certificate(firebase_config) # Assuming firebaseConfig is your service account dict
            initialize_app(cred)
        
        st.session_state.firebase_initialized = True
        db = firestore.client()
        st.session_state.db = db # Store db client in session state
        st.success("Firebase initialized and connected to Firestore.")
except Exception as e:
    st.error(f"Failed to initialize Firebase or connect to Firestore: {e}")
    st.info("Ensure Firebase is correctly configured in your environment.")

def candidate_portal_page():
    """
    Renders the Candidate Portal page.
    Now fetches real application data from Firestore.
    """
    st.markdown('<div class="dashboard-header">🌐 My Candidate Portal</div>', unsafe_allow_html=True)
    
    candidate_email = st.session_state.get('username', 'Guest Candidate')
    st.write(f"Welcome, **{candidate_email}**! Here you can manage your applications.")

    st.markdown("---")

    st.subheader("📄 My Applications")

    # Check if Firestore DB client is available
    if 'db' not in st.session_state:
        st.warning("Firestore connection not established. Cannot load applications.")
        st.info("Please ensure Firebase initialization is successful.")
        return

    db = st.session_state.db
    
    # Fetch real application data from Firestore
    # Assuming applications are stored in a collection named 'candidate_applications'
    # and each document has a 'candidate_email' field matching the logged-in user.
    try:
        # Query applications for the current candidate
        applications_ref = db.collection('candidate_applications').where('candidate_email', '==', candidate_email)
        docs = applications_ref.stream()

        applications_data = []
        for doc in docs:
            app_data = doc.to_dict()
            applications_data.append(app_data)
        
        df_applications = pd.DataFrame(applications_data)

        if not df_applications.empty:
            # Sort by Last Update for better readability
            if 'Last Update' in df_applications.columns:
                df_applications['Last Update'] = pd.to_datetime(df_applications['Last Update'], errors='coerce').dt.date
                df_applications = df_applications.sort_values(by='Last Update', ascending=False)
            
            # Define columns to display and their order
            display_cols = ["Job Title", "Company", "Status", "Last Update", "Feedback"]
            # Ensure only existing columns are displayed
            display_cols = [col for col in display_cols if col in df_applications.columns]

            st.dataframe(df_applications[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("You haven't submitted any applications yet or your applications are not yet linked to this portal.")
            st.write("*(Note: Application data will appear here once your screenings or submissions are recorded in the system.)*")

    except Exception as e:
        st.error(f"Error fetching applications from Firestore: {e}")
        st.info("Please ensure your Firestore security rules allow read access for candidate applications.")


    st.markdown("---")

    st.subheader("⬆️ Upload Updated Documents")
    st.write("You can upload updated versions of your resume or other supporting documents here.")
    uploaded_doc = st.file_uploader("Upload Document (PDF, DOCX)", type=["pdf", "docx"], key="candidate_doc_uploader")
    if uploaded_doc:
        st.success(f"Document '{uploaded_doc.name}' uploaded successfully! (Note: In a full implementation, this would be saved to cloud storage like Google Cloud Storage, and a reference would be stored in Firestore.)")
        # In a real application, you would:
        # 1. Upload the file to a cloud storage bucket (e.g., Google Cloud Storage, AWS S3).
        # 2. Get the public URL or a reference to the stored file.
        # 3. Save this URL/reference in a Firestore document associated with the candidate's application.
        # Example: db.collection('candidate_documents').add({'candidate_email': candidate_email, 'file_name': uploaded_doc.name, 'file_url': 'your_storage_url', 'upload_date': datetime.datetime.now()})

    st.markdown("---")

    st.subheader("🗓️ Schedule Interviews")
    st.info("This section will allow you to view available interview slots and schedule your interviews directly. (Feature under development)")

    st.markdown("---")

    st.subheader("❓ Supplementary Questions")
    st.info("If there are any additional questions or assessments for your application, they will appear here. (Feature under development)")

