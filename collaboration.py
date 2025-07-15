import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin import auth
from google.cloud.firestore import FieldFilter, Query
from datetime import datetime
import json
import os

# --- Firebase Initialization (Safe Check) ---
# Global variables from Canvas environment
app_id = os.environ.get('__app_id', 'default-app-id')
firebase_config_str = os.environ.get('__firebase_config', '{}')
initial_auth_token = os.environ.get('__initial_auth_token', None)

# Parse firebase config
try:
    firebase_config = json.loads(firebase_config_str)
except json.JSONDecodeError:
    st.error("Error parsing Firebase config. Please check the `__firebase_config` variable.")
    firebase_config = {}

# Initialize Firebase Admin SDK if not already initialized
# This part is for server-side operations if needed, but for client-side,
# we primarily use the REST API or client SDK which is handled by the canvas environment.
# However, for `serverTimestamp()`, the Admin SDK is useful.
if not firebase_admin._apps:
    try:
        # Use a service account for Admin SDK if available, otherwise Application Default
        # For Canvas, Application Default Credentials are often set up via GOOGLE_APPLICATION_CREDENTIALS env var
        cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred, name=app_id) # Use app_id as name to avoid re-initialization issues
    except Exception as e:
        st.warning(f"🔥 Firebase Admin SDK init failed: {e}. Some server-side features might be affected.")

# Get Firestore client
db = firestore.client()

# --- Helper function for activity logging (re-used from main.py concept) ---
def log_activity(message):
    """Logs an activity with a timestamp to the session state."""
    if 'activity_log' not in st.session_state:
        st.session_state.activity_log = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.activity_log.insert(0, f"[{timestamp}] {message}") # Add to the beginning for most recent first
    st.session_state.activity_log = st.session_state.activity_log[:50] # Keep log size manageable

def collaboration_hub_page():
    st.markdown('<div class="dashboard-header">🤝 Collaboration Hub</div>', unsafe_allow_html=True)

    st.write("This hub is designed for seamless team collaboration. Share notes, ideas, and updates with your HR team.")

    # Display current user
    current_username = st.session_state.get('username', 'Anonymous User')
    current_user_id = st.session_state.get('user_id', 'N/A') # Assuming user_id is set during Firebase auth
    st.info(f"You are logged in as: **{current_username}** (User ID: `{current_user_id}`)")
    st.markdown("---")

    st.subheader("Shared Team Notes")

    # Input for new note
    new_note_content = st.text_area("Write a new shared note:", key="new_shared_note_input")

    if st.button("Add Note", key="add_note_button"):
        if new_note_content:
            try:
                # Define the collection path for public shared notes
                # This aligns with public data rules: /artifacts/{appId}/public/data/shared_notes
                notes_collection_ref = db.collection(f"artifacts/{app_id}/public/data/shared_notes")

                notes_collection_ref.add({
                    "author": current_username,
                    "content": new_note_content,
                    "timestamp": firestore.SERVER_TIMESTAMP # Use server timestamp for consistency
                })
                st.success("Note added successfully!")
                log_activity(f"User '{current_username}' added a shared note.")
                st.rerun() # Rerun to clear the text area and refresh notes
            except Exception as e:
                st.error(f"Error adding note: {e}")
                log_activity(f"Error adding shared note for '{current_username}': {e}")
        else:
            st.warning("Please write something before adding a note.")

    st.markdown("---")
    st.subheader("Recent Shared Notes")

    # Display existing notes in real-time using on_snapshot
    # Initialize notes list in session state if not present
    if 'shared_notes' not in st.session_state:
        st.session_state.shared_notes = []

    # Set up a real-time listener for shared notes
    # The listener will update st.session_state.shared_notes whenever data changes
    @st.cache_resource(ttl=60) # Cache the listener setup to avoid multiple listeners
    def setup_notes_listener():
        notes_collection_ref = db.collection(f"artifacts/{app_id}/public/data/shared_notes")
        # Order by timestamp descending to show most recent first
        notes_query = notes_collection_ref.order_by("timestamp", direction=Query.DESCENDING).limit(20)

        # Callback function for snapshot listener
        def on_snapshot(col_snapshot, changes, read_time):
            # Process changes to update the session state
            updated_notes = []
            for doc_snapshot in col_snapshot.docs:
                note_data = doc_snapshot.to_dict()
                # Convert timestamp object to string for display
                if 'timestamp' in note_data and hasattr(note_data['timestamp'], 'isoformat'):
                    note_data['timestamp'] = note_data['timestamp'].isoformat()
                updated_notes.append(note_data)
            st.session_state.shared_notes = updated_notes
            # st.rerun() # Rerunning inside callback can cause infinite loops/performance issues.
                        # Streamlit handles state updates and reruns implicitly for widgets.
                        # For non-widget updates, relying on user interaction or periodic refresh is better.

        # Attach the listener
        notes_watcher = notes_query.on_snapshot(on_snapshot)
        # Return the watcher to allow it to be closed if needed (e.g., on logout)
        return notes_watcher

    # Ensure the listener is set up only once
    if 'notes_listener_watcher' not in st.session_state:
        st.session_state.notes_listener_watcher = setup_notes_listener()
        log_activity("Firestore notes listener set up.")

    # Display notes from session state
    if st.session_state.shared_notes:
        for note in st.session_state.shared_notes:
            timestamp_str = note.get('timestamp', 'N/A')
            # Format timestamp if it's a string, otherwise keep as is
            if isinstance(timestamp_str, str) and 'T' in timestamp_str:
                try:
                    dt_object = datetime.fromisoformat(timestamp_str)
                    timestamp_str = dt_object.strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    pass # Keep original string if formatting fails

            st.markdown(f"""
            <div style="background-color: {'#3A3A3A' if st.session_state.get('dark_mode_main') else '#f0f2f6'}; padding: 15px; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <p style="font-size: 0.9em; color: {'#BBBBBB' if st.session_state.get('dark_mode_main') else '#666'}; margin-bottom: 5px;">
                    **{note.get('author', 'Unknown')}** at {timestamp_str}
                </p>
                <p style="font-size: 1.1em; color: {'#E0E0E0' if st.session_state.get('dark_mode_main') else '#333'};">
                    {note.get('content', 'No content')}
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No shared notes yet. Be the first to add one!")

    st.markdown("---")

    st.subheader("Team Member List (Mock)")
    st.info("This section will eventually show active team members and their status.")
    # Mock list of team members
    mock_team_members = [
        {"name": "Alice Johnson", "role": "HR Manager", "status": "Online"},
        {"name": "Bob Williams", "role": "Recruiter", "status": "Online"},
        {"name": "Charlie Brown", "role": "HR Specialist", "status": "Offline"},
    ]
    team_df = pd.DataFrame(mock_team_members)
    st.dataframe(team_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Shared Candidate Pipeline (Mock)")
    st.info("This section will allow teams to manage and track candidates collaboratively.")
    # Mock shared pipeline data
    mock_pipeline = [
        {"Candidate": "Jane Doe", "Status": "Interview Scheduled", "Assigned To": "Alice Johnson"},
        {"Candidate": "John Smith", "Status": "Offer Extended", "Assigned To": "Bob Williams"},
        {"Candidate": "Emily White", "Status": "Screening", "Assigned To": "Charlie Brown"},
    ]
    pipeline_df = pd.DataFrame(mock_pipeline)
    st.dataframe(pipeline_df, use_container_width=True, hide_index=True)

