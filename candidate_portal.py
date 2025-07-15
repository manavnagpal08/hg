import streamlit as st
import pandas as pd
import json
import datetime

# Import Firebase Admin SDK modules
# Make sure to install it: pip install firebase-admin
import firebase_admin
from firebase_admin import credentials, firestore, apps

# --- Firebase Initialization Function ---
# This function centralizes the Firebase setup.
# It's designed to be called once and uses Streamlit's session state to avoid re-initialization.
def initialize_firebase():
    """
    Initializes the Firebase Admin SDK, checking if it's already been done.
    Uses environment variables provided by the execution environment.
    """
    # Check if Firebase is already initialized in the current session
    if not firebase_admin._apps:
        try:
            # Get the app_id and Firebase config from the environment's global variables.
            # These are expected to be injected by the hosting environment.
            app_id = globals().get('__app_id', 'default-app-id')
            firebase_config_str = globals().get('__firebase_config', '{}')
            
            # If the config string is not empty, parse it.
            if firebase_config_str and firebase_config_str != '{}':
                firebase_config = json.loads(firebase_config_str)
                cred = credentials.Certificate(firebase_config)
                firebase_admin.initialize_app(cred)
                st.session_state.firebase_initialized = True
                st.success("Firebase initialized successfully using provided configuration!")
            else:
                # Fallback for environments where default credentials are set up
                # (e.g., Google Cloud Run, App Engine)
                firebase_admin.initialize_app()
                st.session_state.firebase_initialized = True
                st.success("Firebase initialized successfully using default credentials!")

            # Store the app_id and Firestore client in the session state for later use
            st.session_state.app_id = app_id
            st.session_state.db = firestore.client()

        except json.JSONDecodeError:
            st.error("Firebase configuration is not a valid JSON string. Please check the environment setup.")
            st.session_state.firebase_initialized = False
        except Exception as e:
            st.error(f"Failed to initialize Firebase: {e}")
            st.info("Please ensure your Firebase configuration and credentials are set up correctly in the environment.")
            st.session_state.firebase_initialized = False
    
    # Return the initialization status
    return st.session_state.get('firebase_initialized', False)


def candidate_portal_page():
    """
    Renders the Candidate Portal page.
    Fetches real application data from Firestore after ensuring Firebase is initialized.
    """
    st.markdown('<div class="dashboard-header">🌐 My Candidate Portal</div>', unsafe_allow_html=True)
    
    # Get candidate email from session state (assuming it's set on login)
    candidate_email = st.session_state.get('username', 'guest.candidate@example.com')
    st.write(f"Welcome, **{candidate_email}**! Here you can manage your applications.")

    st.markdown("---")

    st.subheader("📄 My Applications")

    # --- Data Fetching Logic ---
    # First, ensure Firebase is ready.
    if not initialize_firebase():
        st.warning("Firestore connection is not available because Firebase initialization failed.")
        return # Stop execution of this part if DB is not ready

    db = st.session_state.db
    app_id = st.session_state.app_id
    
    try:
        # Construct the correct collection path using the app_id
        collection_path = f'artifacts/{app_id}/public/data/candidate_applications'
        
        # Query applications for the current candidate
        applications_ref = db.collection(collection_path).where('candidate_email', '==', candidate_email)
        docs = applications_ref.stream()

        applications_data = []
        for doc in docs:
            app_data = doc.to_dict()
            app_data['doc_id'] = doc.id # Keep the document ID if needed
            applications_data.append(app_data)
        
        if not applications_data:
            st.info("You haven't submitted any applications yet or they are not yet linked to this portal.")
            st.write("*(Note: Application data will appear here once submissions are recorded.)*")
        else:
            df_applications = pd.DataFrame(applications_data)
            
            # --- Data Display and Formatting ---
            # Standardize 'Last Update' column for reliable sorting
            if 'Last Update' in df_applications.columns:
                df_applications['Last Update'] = pd.to_datetime(df_applications['Last Update'], errors='coerce').dt.date
                df_applications = df_applications.sort_values(by='Last Update', ascending=False)
            
            # Define columns to display and their order
            display_cols = ["Job Title", "Company", "Status", "Last Update", "Feedback"]
            # Filter out columns that don't exist in the DataFrame to prevent errors
            existing_display_cols = [col for col in display_cols if col in df_applications.columns]

            st.dataframe(df_applications[existing_display_cols], use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error fetching applications from Firestore: {e}")
        st.info("Please ensure your Firestore security rules allow read access for this path and that the collection exists.")

    st.markdown("---")

    # --- Other Page Sections (Unchanged) ---
    st.subheader("⬆️ Upload Updated Documents")
    st.write("You can upload updated versions of your resume or other supporting documents here.")
    uploaded_doc = st.file_uploader("Upload Document (PDF, DOCX)", type=["pdf", "docx"], key="candidate_doc_uploader")
    if uploaded_doc:
        st.success(f"Document '{uploaded_doc.name}' uploaded successfully! (Note: This is a demo. The file is not saved.)")

    st.markdown("---")
    st.subheader("🗓️ Schedule Interviews")
    st.info("This section will allow you to view available interview slots and schedule your interviews directly. (Feature under development)")

    st.markdown("---")
    st.subheader("❓ Supplementary Questions")
    st.info("If there are any additional questions or assessments for your application, they will appear here. (Feature under development)")

# --- To run this page ---
# You would typically call this function from your main Streamlit app file based on user navigation.
# For demonstration purposes, you can run it directly.
if __name__ == '__main__':
    # Mock username for direct script execution
    if 'username' not in st.session_state:
        st.session_state.username = 'test.user@example.com'
    
    candidate_portal_page()

