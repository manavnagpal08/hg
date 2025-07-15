import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin import auth
from google.cloud.firestore import FieldFilter, Query
from datetime import datetime
import json
import os
import pandas as pd # Import pandas for DataFrame

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
        firebase_admin.initialize_app(cred, name=app_id)
    except Exception as e:
        st.warning(f"🔥 Firebase Admin SDK init failed: {e}. Some server-side features might be affected.")

# Get Firestore client
db = firestore.client()

# --- Helper function for activity logging (re-used from main.py concept) ---
def log_activity(message: str, user: str = None):
    """
    Logs an activity with a timestamp to Firestore and session state.
    This helps track user actions and application events.
    """
    if user is None:
        user = st.session_state.get('username', 'Anonymous User')

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {user}: {message}"

    # Add to session state log (for immediate display in the Activity Feed tab)
    if 'activity_log' not in st.session_state:
        st.session_state.activity_log = []
    st.session_state.activity_log.insert(0, log_entry) # Add to the beginning for most recent first
    st.session_state.activity_log = st.session_state.activity_log[:50] # Keep log size manageable

    # Persist to Firestore for long-term storage and cross-session visibility
    try:
        activity_collection_ref = db.collection(f"artifacts/{app_id}/public/data/activity_feed")
        activity_collection_ref.add({
            "message": message,
            "user": user,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        st.error(f"Error logging activity to Firestore: {e}")
        # IMPORTANT: If this error persists, double-check your Firestore security rules for the 'activity_feed' collection.
        # Ensure it allows authenticated users to write, e.g., 'allow read, write: if request.auth != null;'

def collaboration_hub_page():
    st.markdown('<div class="dashboard-header">🤝 Collaboration Hub</div>', unsafe_allow_html=True)

    st.write("This hub is designed for seamless team collaboration. Share notes, ideas, and updates with your HR team.")

    # Display current user information
    current_username = st.session_state.get('username', 'Anonymous User')
    # Use current_username as a unique identifier for Firestore paths in this context.
    # In a full Firebase Auth setup, `auth.currentUser.uid` would be the robust user ID.
    current_user_id = st.session_state.get('user_id', current_username)
    st.info(f"You are logged in as: **{current_username}** (User ID: `{current_user_id}`)")
    st.markdown("---")

    # --- Tabs for different collaboration features ---
    tab_notes, tab_tasks, tab_announcements, tab_members, tab_messages, tab_files, tab_calendar, tab_resources, tab_polls, tab_activity, tab_pipeline = st.tabs([
        "📝 Shared Notes", "✅ Team Tasks", "📢 Announcements", "👥 Team Members", "💬 Direct Messages (Mock)", "📂 File Sharing (Mock)",
        "🗓️ Team Calendar", "📚 Resource Library", "📊 Quick Polls", "⚡ Activity Feed", "📋 Candidate Pipeline (Mock)"
    ])

    with tab_notes:
        st.subheader("Shared Team Notes")

        # Input for new note
        new_note_content = st.text_area("Write a new shared note:", key="new_shared_note_input")

        if st.button("Add Note", key="add_note_button"):
            if new_note_content:
                try:
                    # Define the collection path for public shared notes
                    # Firestore security rules for this collection should allow:
                    # allow read, write: if request.auth != null;
                    notes_collection_ref = db.collection(f"artifacts/{app_id}/public/data/shared_notes")

                    notes_collection_ref.add({
                        "author": current_username,
                        "content": new_note_content,
                        "timestamp": firestore.SERVER_TIMESTAMP # Use server timestamp for consistency
                    })
                    st.success("Note added successfully!")
                    log_activity(f"added a shared note.", user=current_username)
                    st.rerun() # Rerun to clear the text area and refresh notes
                except Exception as e:
                    st.error(f"Error adding note: {e}")
                    log_activity(f"Error adding shared note: {e}", user=current_username)
            else:
                st.warning("Please write something before adding a note.")

        st.markdown("---")
        st.subheader("Recent Shared Notes")

        # Display existing notes in real-time using on_snapshot
        # Initialize notes list in session state if not present
        if 'shared_notes' not in st.session_state:
            st.session_state.shared_notes = []

        # Set up a real-time listener for shared notes
        # @st.cache_resource ensures this setup function runs only once per session,
        # preventing multiple listeners from being attached. The on_snapshot callback
        # itself provides real-time updates to st.session_state.
        @st.cache_resource(ttl=60) # TTL (Time To Live) in seconds for the cache
        def setup_notes_listener():
            notes_collection_ref = db.collection(f"artifacts/{app_id}/public/data/shared_notes")
            # Order by timestamp descending to show most recent first
            notes_query = notes_collection_ref.order_by("timestamp", direction=Query.DESCENDING).limit(20)

            # Callback function for snapshot listener: updates session state when data changes
            def on_snapshot(col_snapshot, changes, read_time):
                updated_notes = []
                for doc_snapshot in col_snapshot.docs:
                    note_data = doc_snapshot.to_dict()
                    # Convert Firestore Timestamp object to string for display
                    if 'timestamp' in note_data and hasattr(note_data['timestamp'], 'isoformat'):
                        note_data['timestamp'] = note_data['timestamp'].isoformat()
                    updated_notes.append(note_data)
                st.session_state.shared_notes = updated_notes

            notes_watcher = notes_query.on_snapshot(on_snapshot)
            return notes_watcher

        # Ensure the listener is set up only once per Streamlit session
        if 'notes_listener_watcher' not in st.session_state:
            st.session_state.notes_listener_watcher = setup_notes_listener()
            log_activity("Firestore notes listener set up.", user="System")

        # Display notes from session state
        if st.session_state.shared_notes:
            for note in st.session_state.shared_notes:
                timestamp_str = note.get('timestamp', 'N/A')
                # Re-format timestamp string for better readability if needed
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

    with tab_tasks:
        st.subheader("Team Task Management")
        st.write("Assign and track tasks for your team.")

        new_task_description = st.text_input("New Task Description:", key="new_task_input")
        new_task_assignee = st.text_input("Assignee (e.g., 'Alice', 'Bob', 'All'):", key="new_task_assignee")

        if st.button("Add Task", key="add_task_button"):
            if new_task_description and new_task_assignee:
                try:
                    # Firestore security rules for this collection should allow:
                    # allow read, write: if request.auth != null;
                    tasks_collection_ref = db.collection(f"artifacts/{app_id}/public/data/team_tasks")
                    tasks_collection_ref.add({
                        "description": new_task_description,
                        "assignee": new_task_assignee,
                        "status": "pending", # Initial status
                        "created_by": current_username,
                        "created_at": firestore.SERVER_TIMESTAMP
                    })
                    st.success("Task added successfully!")
                    log_activity(f"added a new task: '{new_task_description}'.", user=current_username)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding task: {e}")
                    log_activity(f"Error adding task: {e}", user=current_username)
            else:
                st.warning("Please provide a task description and assignee.")

        st.markdown("---")
        st.subheader("Current Tasks")

        if 'team_tasks' not in st.session_state:
            st.session_state.team_tasks = []

        @st.cache_resource(ttl=60)
        def setup_tasks_listener():
            tasks_collection_ref = db.collection(f"artifacts/{app_id}/public/data/team_tasks")
            tasks_query = tasks_collection_ref.order_by("created_at", direction=Query.DESCENDING).limit(20)

            def on_snapshot(col_snapshot, changes, read_time):
                updated_tasks = []
                for doc_snapshot in col_snapshot.docs:
                    task_data = doc_snapshot.to_dict()
                    task_data['id'] = doc_snapshot.id # Store document ID for updates
                    if 'created_at' in task_data and hasattr(task_data['created_at'], 'isoformat'):
                        task_data['created_at'] = task_data['created_at'].isoformat()
                    updated_tasks.append(task_data)
                st.session_state.team_tasks = updated_tasks

            tasks_watcher = tasks_query.on_snapshot(on_snapshot)
            return tasks_watcher

        if 'tasks_listener_watcher' not in st.session_state:
            st.session_state.tasks_listener_watcher = setup_tasks_listener()
            log_activity("Firestore tasks listener set up.", user="System")

        if st.session_state.team_tasks:
            for task in st.session_state.team_tasks:
                timestamp_str = task.get('created_at', 'N/A')
                if isinstance(timestamp_str, str) and 'T' in timestamp_str:
                    try:
                        dt_object = datetime.fromisoformat(timestamp_str)
                        timestamp_str = dt_object.strftime("%Y-%m-%d %H:%M")
                    except ValueError:
                        pass

                status_color = "green" if task.get('status') == "completed" else "orange"
                st.markdown(f"""
                <div style="background-color: {'#3A3A3A' if st.session_state.get('dark_mode_main') else '#f0f2f6'}; padding: 15px; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <p style="font-size: 0.9em; color: {'#BBBBBB' if st.session_state.get('dark_mode_main') else '#666'}; margin-bottom: 5px;">
                        Assigned to: **{task.get('assignee', 'N/A')}** | Status: <span style="color: {status_color}; font-weight: bold;">{task.get('status', 'N/A').upper()}</span>
                        (Created by: {task.get('created_by', 'Unknown')} at {timestamp_str})
                    </p>
                    <p style="font-size: 1.1em; color: {'#E0E0E0' if st.session_state.get('dark_mode_main') else '#333'};">
                        {task.get('description', 'No description')}
                    </p>
                """, unsafe_allow_html=True)
                # Only show "Mark as Complete" button if the task is pending
                if task.get('status') == 'pending':
                    if st.button(f"Mark as Complete", key=f"complete_task_{task['id']}"):
                        try:
                            tasks_collection_ref = db.collection(f"artifacts/{app_id}/public/data/team_tasks")
                            tasks_collection_ref.document(task['id']).update({"status": "completed"})
                            st.success("Task marked as complete!")
                            log_activity(f"marked task '{task['description']}' as complete.", user=current_username)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error updating task: {e}")
                            log_activity(f"Error updating task '{task['description']}': {e}", user=current_username)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No tasks added yet. Add a new task above!")

    with tab_announcements:
        st.subheader("Team Announcements")
        st.write("Post important announcements for the entire team.")

        new_announcement_title = st.text_input("Announcement Title:", key="new_announcement_title")
        new_announcement_content = st.text_area("Announcement Details:", key="new_announcement_content")

        if st.button("Post Announcement", key="post_announcement_button"):
            if new_announcement_title and new_announcement_content:
                try:
                    # Firestore security rules for this collection should allow:
                    # allow read, write: if request.auth != null;
                    announcements_collection_ref = db.collection(f"artifacts/{app_id}/public/data/team_announcements")
                    announcements_collection_ref.add({
                        "title": new_announcement_title,
                        "content": new_announcement_content,
                        "author": current_username,
                        "timestamp": firestore.SERVER_TIMESTAMP
                    })
                    st.success("Announcement posted successfully!")
                    log_activity(f"posted an announcement: '{new_announcement_title}'.", user=current_username)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error posting announcement: {e}")
                    log_activity(f"Error posting announcement: {e}", user=current_username)
            else:
                st.warning("Please provide both a title and content for the announcement.")

        st.markdown("---")
        st.subheader("Recent Announcements")

        if 'team_announcements' not in st.session_state:
            st.session_state.team_announcements = []

        @st.cache_resource(ttl=60)
        def setup_announcements_listener():
            announcements_collection_ref = db.collection(f"artifacts/{app_id}/public/data/team_announcements")
            announcements_query = announcements_collection_ref.order_by("timestamp", direction=Query.DESCENDING).limit(10)

            def on_snapshot(col_snapshot, changes, read_time):
                updated_announcements = []
                for doc_snapshot in col_snapshot.docs:
                    announcement_data = doc_snapshot.to_dict()
                    if 'timestamp' in announcement_data and hasattr(announcement_data['timestamp'], 'isoformat'):
                        announcement_data['timestamp'] = announcement_data['timestamp'].isoformat()
                    updated_announcements.append(announcement_data)
                st.session_state.team_announcements = updated_announcements

            announcements_watcher = announcements_query.on_snapshot(on_snapshot)
            return announcements_watcher

        if 'announcements_listener_watcher' not in st.session_state:
            st.session_state.announcements_listener_watcher = setup_announcements_listener()
            log_activity("Firestore announcements listener set up.", user="System")

        if st.session_state.team_announcements:
            for announcement in st.session_state.team_announcements:
                timestamp_str = announcement.get('timestamp', 'N/A')
                if isinstance(timestamp_str, str) and 'T' in timestamp_str:
                    try:
                        dt_object = datetime.fromisoformat(timestamp_str)
                        timestamp_str = dt_object.strftime("%Y-%m-%d %H:%M")
                    except ValueError:
                        pass

                st.markdown(f"""
                <div style="background-color: {'#3A3A3A' if st.session_state.get('dark_mode_main') else '#f0f2f6'}; padding: 15px; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <h4 style="color: {'#00cec9' if st.session_state.get('dark_mode_main') else '#00cec9'}; margin-bottom: 5px;">{announcement.get('title', 'No Title')}</h4>
                    <p style="font-size: 0.9em; color: {'#BBBBBB' if st.session_state.get('dark_mode_main') else '#666'}; margin-bottom: 5px;">
                        Posted by: **{announcement.get('author', 'Unknown')}** at {timestamp_str}
                    </p>
                    <p style="font-size: 1.0em; color: {'#E0E0E0' if st.session_state.get('dark_mode_main') else '#333'};">
                        {announcement.get('content', 'No content')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No announcements posted yet. Be the first to share important updates!")

    with tab_members:
        st.subheader("Team Member Management")
        st.write("View current team members and add new ones to your organization.")

        # Form to add new team members
        with st.form("add_team_member_form", clear_on_submit=True):
            new_member_name = st.text_input("New Member's Name:", key="new_member_name_input")
            new_member_email = st.text_input("New Member's Email (will be their User ID):", key="new_member_email_input")
            new_member_role = st.text_input("New Member's Role (e.g., 'Recruiter', 'HR Specialist'):", key="new_member_role_input")
            add_member_button = st.form_submit_button("Add Team Member")

            if add_member_button:
                if new_member_name and new_member_email and new_member_role:
                    # Basic email format validation
                    if "@" not in new_member_email or "." not in new_member_email:
                        st.error("Please enter a valid email address for the new member.")
                    else:
                        try:
                            # Store team member profiles in a public collection
                            # Firestore security rules: allow read, write: if request.auth != null;
                            team_members_collection_ref = db.collection(f"artifacts/{app_id}/public/data/team_members")
                            
                            # Check if a member with this email already exists
                            existing_member = team_members_collection_ref.document(new_member_email).get()
                            if existing_member.exists:
                                st.warning(f"A team member with email '{new_member_email}' already exists.")
                            else:
                                team_members_collection_ref.document(new_member_email).set({
                                    "name": new_member_name,
                                    "email": new_member_email,
                                    "role": new_member_role,
                                    "status": "active", # Default status
                                    "added_by": current_username,
                                    "added_at": firestore.SERVER_TIMESTAMP
                                })
                                st.success(f"Team member '{new_member_name}' added successfully!")
                                log_activity(f"added new team member: '{new_member_name}' ({new_member_email}).", user=current_username)
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error adding team member: {e}")
                            log_activity(f"Error adding team member: {e}", user=current_username)
                else:
                    st.warning("Please fill in all fields for the new team member.")

        st.markdown("---")
        st.subheader("Current Team Members")

        if 'team_members' not in st.session_state:
            st.session_state.team_members = []

        @st.cache_resource(ttl=60)
        def setup_team_members_listener():
            team_members_collection_ref = db.collection(f"artifacts/{app_id}/public/data/team_members")
            team_members_query = team_members_collection_ref.order_by("added_at", direction=Query.ASCENDING).limit(50)

            def on_snapshot(col_snapshot, changes, read_time):
                updated_members = []
                for doc_snapshot in col_snapshot.docs:
                    member_data = doc_snapshot.to_dict()
                    if 'added_at' in member_data and hasattr(member_data['added_at'], 'isoformat'):
                        member_data['added_at'] = member_data['added_at'].isoformat()
                    updated_members.append(member_data)
                st.session_state.team_members = updated_members

            team_members_watcher = team_members_query.on_snapshot(on_snapshot)
            return team_members_watcher

        if 'team_members_listener_watcher' not in st.session_state:
            st.session_state.team_members_listener_watcher = setup_team_members_listener()
            log_activity("Firestore team members listener set up.", user="System")

        if st.session_state.team_members:
            team_members_df = pd.DataFrame(st.session_state.team_members)
            # Display only relevant columns for the table
            display_cols = ['name', 'email', 'role', 'status', 'added_by']
            st.dataframe(team_members_df[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No team members added yet. Add new members using the form above!")


    with tab_messages:
        st.subheader("Direct Messages (Mock)")
        st.info("This section is a placeholder for direct, private messaging between team members. A full implementation would require a more complex backend for secure, real-time communication.")
        st.write("For now, imagine sending a message to a specific team member:")

        # Populate recipient list from actual team members if available, otherwise use mock
        if st.session_state.get('team_members'):
            recipient_options = [member['email'] for member in st.session_state.team_members]
            if current_username in recipient_options:
                recipient_options.remove(current_username) # Don't allow messaging self
            if not recipient_options:
                recipient_options = ["No other members available"]
        else:
            recipient_options = ["Alice Johnson (mock)", "Bob Williams (mock)", "Charlie Brown (mock)"]

        mock_recipient = st.selectbox("Select Recipient", recipient_options, key="mock_dm_recipient")
        mock_message = st.text_area(f"Message to {mock_recipient}:", key="mock_dm_message")

        if st.button("Send Mock Message", key="send_mock_dm_button"):
            if mock_message and mock_recipient != "No other members available":
                st.success(f"Mock message sent to {mock_recipient}: '{mock_message}'")
                log_activity(f"sent a mock DM to '{mock_recipient}'.", user=current_username)
            else:
                st.warning("Please type a message and select a valid recipient to send.")

        st.markdown("---")
        st.subheader("Your Mock Message History")
        st.info("Your sent and received direct messages would appear here in a real implementation.")
        st.write("*(No actual messages are stored or sent in this mock feature)*")

    with tab_files:
        st.subheader("Shared Files (Mock)")
        st.info("This section is a placeholder for secure file sharing among your HR team. A full implementation would require integration with cloud storage (e.g., Firebase Storage) and robust security measures.")
        st.write("You will be able to upload, download, and manage shared documents here.")
        st.markdown("---")
        st.subheader("Recently Shared Files")
        st.write("*(No files available in this mock feature)*")

    with tab_calendar:
        st.subheader("Team Calendar")
        st.write("Add and view important team events, interview schedules, and deadlines.")

        with st.form("add_event_form", clear_on_submit=True):
            event_title = st.text_input("Event Title:", key="event_title_input")
            event_date = st.date_input("Event Date:", key="event_date_input")
            event_time = st.text_input("Event Time (e.g., 10:00 AM, 14:30):", key="event_time_input")
            event_description = st.text_area("Event Description:", key="event_description_input")
            add_event_button = st.form_submit_button("Add Event")

            if add_event_button:
                if event_title and event_date and event_time:
                    try:
                        # Firestore security rules for this collection should allow:
                        # allow read, write: if request.auth != null;
                        events_collection_ref = db.collection(f"artifacts/{app_id}/public/data/team_events")
                        events_collection_ref.add({
                            "title": event_title,
                            "date": str(event_date), # Store date as string (YYYY-MM-DD)
                            "time": event_time,
                            "description": event_description,
                            "created_by": current_username,
                            "created_at": firestore.SERVER_TIMESTAMP
                        })
                        st.success("Event added successfully!")
                        log_activity(f"added a new calendar event: '{event_title}'.", user=current_username)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding event: {e}")
                        log_activity(f"Error adding event: {e}", user=current_username)
                else:
                    st.warning("Please fill in event title, date, and time.")

        st.markdown("---")
        st.subheader("Upcoming Events")

        if 'team_events' not in st.session_state:
            st.session_state.team_events = []

        @st.cache_resource(ttl=60)
        def setup_events_listener():
            events_collection_ref = db.collection(f"artifacts/{app_id}/public/data/team_events")
            # Order by date and then time to show upcoming events correctly
            events_query = events_collection_ref.order_by("date", direction=Query.ASCENDING).order_by("time", direction=Query.ASCENDING).limit(10)

            def on_snapshot(col_snapshot, changes, read_time):
                updated_events = []
                for doc_snapshot in col_snapshot.docs:
                    event_data = doc_snapshot.to_dict()
                    if 'created_at' in event_data and hasattr(event_data['created_at'], 'isoformat'):
                        event_data['created_at'] = event_data['created_at'].isoformat()
                    updated_events.append(event_data)
                st.session_state.team_events = updated_events

            events_watcher = events_query.on_snapshot(on_snapshot)
            return events_watcher

        if 'events_listener_watcher' not in st.session_state:
            st.session_state.events_listener_watcher = setup_events_listener()
            log_activity("Firestore events listener set up.", user="System")

        if st.session_state.team_events:
            # Filter out past events for "Upcoming Events" display
            today = datetime.now().date()
            upcoming_events = [
                event for event in st.session_state.team_events
                if datetime.strptime(event.get('date', '1900-01-01'), '%Y-%m-%d').date() >= today
            ]
            if upcoming_events:
                for event in upcoming_events:
                    st.markdown(f"""
                    <div style="background-color: {'#3A3A3A' if st.session_state.get('dark_mode_main') else '#f0f2f6'}; padding: 15px; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                        <h4 style="color: {'#00cec9' if st.session_state.get('dark_mode_main') else '#00cec9'}; margin-bottom: 5px;">{event.get('title', 'No Title')}</h4>
                        <p style="font-size: 0.9em; color: {'#BBBBBB' if st.session_state.get('dark_mode_main') else '#666'}; margin-bottom: 5px;">
                            Date: **{event.get('date', 'N/A')}** | Time: **{event.get('time', 'N/A')}**
                        </p>
                        <p style="font-size: 1.0em; color: {'#E0E0E0' if st.session_state.get('dark_mode_main') else '#333'};">
                            {event.get('description', 'No description')}
                        </p>
                        <p style="font-size: 0.8em; color: {'#999999' if st.session_state.get('dark_mode_main') else '#888'}; margin-top: 5px;">
                            Added by: {event.get('created_by', 'Unknown')}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No upcoming events scheduled.")
        else:
            st.info("No events added yet. Add a new event above!")

    with tab_resources:
        st.subheader("Resource Library")
        st.write("Manage and access important HR documents, templates, and guidelines.")

        with st.form("add_resource_form", clear_on_submit=True):
            resource_name = st.text_input("Resource Name:", key="resource_name_input")
            resource_url = st.text_input("Resource URL (if applicable):", key="resource_url_input")
            resource_description = st.text_area("Description:", key="resource_description_input")
            resource_category = st.selectbox("Category:", ["Policy", "Template", "Guide", "Training", "Other"], key="resource_category_select")
            add_resource_button = st.form_submit_button("Add Resource")

            if add_resource_button:
                if resource_name:
                    try:
                        # Firestore security rules for this collection should allow:
                        # allow read, write: if request.auth != null;
                        resources_collection_ref = db.collection(f"artifacts/{app_id}/public/data/resource_library")
                        resources_collection_ref.add({
                            "name": resource_name,
                            "url": resource_url,
                            "description": resource_description,
                            "category": resource_category,
                            "uploaded_by": current_username,
                            "uploaded_at": firestore.SERVER_TIMESTAMP
                        })
                        st.success("Resource added successfully!")
                        log_activity(f"added a new resource: '{resource_name}'.", user=current_username)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding resource: {e}")
                        log_activity(f"Error adding resource: {e}", user=current_username)
                else:
                    st.warning("Please provide a resource name.")

        st.markdown("---")
        st.subheader("Available Resources")

        if 'resource_library' not in st.session_state:
            st.session_state.resource_library = []

        @st.cache_resource(ttl=60)
        def setup_resources_listener():
            resources_collection_ref = db.collection(f"artifacts/{app_id}/public/data/resource_library")
            resources_query = resources_collection_ref.order_by("uploaded_at", direction=Query.DESCENDING).limit(20)

            def on_snapshot(col_snapshot, changes, read_time):
                updated_resources = []
                for doc_snapshot in col_snapshot.docs:
                    resource_data = doc_snapshot.to_dict()
                    if 'uploaded_at' in resource_data and hasattr(resource_data['uploaded_at'], 'isoformat'):
                        resource_data['uploaded_at'] = resource_data['uploaded_at'].isoformat()
                    updated_resources.append(resource_data)
                st.session_state.resource_library = updated_resources

            resources_watcher = resources_query.on_snapshot(on_snapshot)
            return resources_watcher

        if 'resources_listener_watcher' not in st.session_state:
            st.session_state.resources_listener_watcher = setup_resources_listener()
            log_activity("Firestore resources listener set up.", user="System")

        if st.session_state.resource_library:
            for resource in st.session_state.resource_library:
                timestamp_str = resource.get('uploaded_at', 'N/A')
                if isinstance(timestamp_str, str) and 'T' in timestamp_str:
                    try:
                        dt_object = datetime.fromisoformat(timestamp_str)
                        timestamp_str = dt_object.strftime("%Y-%m-%d %H:%M")
                    except ValueError:
                        pass

                st.markdown(f"""
                <div style="background-color: {'#3A3A3A' if st.session_state.get('dark_mode_main') else '#f0f2f6'}; padding: 15px; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <h4 style="color: {'#00cec9' if st.session_state.get('dark_mode_main') else '#00cec9'}; margin-bottom: 5px;">
                        {resource.get('name', 'No Name')} ({resource.get('category', 'N/A')})
                    </h4>
                    <p style="font-size: 1.0em; color: {'#E0E0E0' if st.session_state.get('dark_mode_main') else '#333'};">
                        {resource.get('description', 'No description')}
                    </p>
                    {"<p><a href='" + resource['url'] + "' target='_blank' style='color: #00cec9;'>View Resource</a></p>" if resource.get('url') else ""}
                    <p style="font-size: 0.8em; color: {'#999999' if st.session_state.get('dark_mode_main') else '#888'}; margin-top: 5px;">
                        Uploaded by: {resource.get('uploaded_by', 'Unknown')} at {timestamp_str}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No resources added yet. Add a new resource above!")

    with tab_polls:
        st.subheader("Quick Polls/Surveys")
        st.write("Create and participate in quick polls to gather team feedback.")

        with st.form("create_poll_form", clear_on_submit=True):
            poll_question = st.text_input("Poll Question:", key="poll_question_input")
            poll_options_str = st.text_area("Options (comma-separated):", key="poll_options_input")
            create_poll_button = st.form_submit_button("Create New Poll")

            if create_poll_button:
                if poll_question and poll_options_str:
                    options = [opt.strip() for opt in poll_options_str.split(',') if opt.strip()]
                    if len(options) < 2:
                        st.warning("Please provide at least two options for the poll.")
                    else:
                        try:
                            # Firestore security rules for this collection should allow:
                            # allow read, write: if request.auth != null;
                            polls_collection_ref = db.collection(f"artifacts/{app_id}/public/data/team_polls")
                            polls_collection_ref.add({
                                "question": poll_question,
                                "options": options,
                                "votes": {option: 0 for option in options}, # Initialize votes to 0
                                "active": True,
                                "created_by": current_username,
                                "created_at": firestore.SERVER_TIMESTAMP
                            })
                            st.success("Poll created successfully!")
                            log_activity(f"created a new poll: '{poll_question}'.", user=current_username)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error creating poll: {e}")
                            log_activity(f"Error creating poll: {e}", user=current_username)
                else:
                    st.warning("Please provide a poll question and options.")

        st.markdown("---")
        st.subheader("Active Polls")

        if 'active_polls' not in st.session_state:
            st.session_state.active_polls = []

        @st.cache_resource(ttl=60)
        def setup_polls_listener():
            polls_collection_ref = db.collection(f"artifacts/{app_id}/public/data/team_polls")
            # Only fetch active polls
            polls_query = polls_collection_ref.where(filter=FieldFilter("active", "==", True)).order_by("created_at", direction=Query.DESCENDING).limit(5)

            def on_snapshot(col_snapshot, changes, read_time):
                updated_polls = []
                for doc_snapshot in col_snapshot.docs:
                    poll_data = doc_snapshot.to_dict()
                    poll_data['id'] = doc_snapshot.id # Store document ID
                    if 'created_at' in poll_data and hasattr(poll_data['created_at'], 'isoformat'):
                        poll_data['created_at'] = poll_data['created_at'].isoformat()
                    updated_polls.append(poll_data)
                st.session_state.active_polls = updated_polls

            polls_watcher = polls_query.on_snapshot(on_snapshot)
            return polls_watcher

        if 'polls_listener_watcher' not in st.session_state:
            st.session_state.polls_listener_watcher = setup_polls_listener()
            log_activity("Firestore polls listener set up.", user="System")

        if st.session_state.active_polls:
            for poll in st.session_state.active_polls:
                st.markdown(f"""
                <div style="background-color: {'#3A3A3A' if st.session_state.get('dark_mode_main') else '#f0f2f6'}; padding: 15px; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <h4 style="color: {'#00cec9' if st.session_state.get('dark_mode_main') else '#00cec9'}; margin-bottom: 5px;">{poll.get('question', 'No Question')}</h4>
                    <p style="font-size: 0.9em; color: {'#BBBBBB' if st.session_state.get('dark_mode_main') else '#666'}; margin-bottom: 5px;">
                        Created by: **{poll.get('created_by', 'Unknown')}**
                    </p>
                """, unsafe_allow_html=True)

                # Check if user has already voted in this session
                user_voted_key = f"voted_poll_{poll['id']}_{current_username}"
                if st.session_state.get(user_voted_key):
                    st.info("You have already voted on this poll in this session.")
                else:
                    selected_option = st.radio("Select your choice:", poll.get('options', []), key=f"poll_option_{poll['id']}")
                    if st.button("Submit Vote", key=f"submit_vote_{poll['id']}"):
                        if selected_option:
                            try:
                                poll_ref = db.collection(f"artifacts/{app_id}/public/data/team_polls").document(poll['id'])
                                # Increment the vote count for the selected option.
                                # For production, consider using Firestore transactions (db.transaction)
                                # or FieldValue.increment for safer concurrent updates.
                                # Example: poll_ref.update({f"votes.{selected_option}": firestore.Increment(1)})
                                current_votes = poll.get('votes', {})
                                current_votes[selected_option] = current_votes.get(selected_option, 0) + 1
                                poll_ref.update({"votes": current_votes})
                                st.session_state[user_voted_key] = True # Mark user as voted in session
                                st.success("Vote submitted successfully!")
                                log_activity(f"voted on poll '{poll['question']}' for option '{selected_option}'.", user=current_username)
                                st.rerun() # Rerun to update the displayed results
                            except Exception as e:
                                st.error(f"Error submitting vote: {e}")
                                log_activity(f"Error submitting vote: {e}", user=current_username)
                        else:
                            st.warning("Please select an option to vote.")
                
                # Display results
                st.markdown("##### Current Results:")
                votes_data = poll.get('votes', {})
                if votes_data:
                    results_df = pd.DataFrame(list(votes_data.items()), columns=['Option', 'Votes']).set_index('Option')
                    st.bar_chart(results_df)
                else:
                    st.info("No votes yet.")

                # Option to close poll (only for creator or admin)
                admin_usernames = ("admin@forscreenerpro", "admin@forscreenerpro2", "manav.nagpal2005@gmail.2005@gmail.com")
                if current_username == poll.get('created_by') or current_username in admin_usernames:
                    if poll.get('active') and st.button("Close Poll", key=f"close_poll_{poll['id']}"):
                        try:
                            poll_ref = db.collection(f"artifacts/{app_id}/public/data/team_polls").document(poll['id'])
                            poll_ref.update({"active": False})
                            st.success("Poll closed successfully!")
                            log_activity(f"closed poll: '{poll['question']}'.", user=current_username)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error closing poll: {e}")
                            log_activity(f"Error closing poll: {e}", user=current_username)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No active polls. Be the first to create one!")

    with tab_activity:
        st.subheader("Recent Activity Feed")
        st.write("See a chronological log of recent actions within the Collaboration Hub across all users.")

        if 'firestore_activity_log' not in st.session_state:
            st.session_state.firestore_activity_log = []

        @st.cache_resource(ttl=60)
        def setup_activity_feed_listener():
            activity_collection_ref = db.collection(f"artifacts/{app_id}/public/data/activity_feed")
            activity_query = activity_collection_ref.order_by("timestamp", direction=Query.DESCENDING).limit(30)

            def on_snapshot(col_snapshot, changes, read_time):
                updated_activity = []
                for doc_snapshot in col_snapshot.docs:
                    activity_data = doc_snapshot.to_dict()
                    if 'timestamp' in activity_data and hasattr(activity_data['timestamp'], 'isoformat'):
                        activity_data['timestamp'] = activity_data['timestamp'].isoformat()
                    updated_activity.append(activity_data)
                st.session_state.firestore_activity_log = updated_activity

            activity_watcher = activity_query.on_snapshot(on_snapshot)
            return activity_watcher

        if 'activity_feed_listener_watcher' not in st.session_state:
            st.session_state.activity_feed_listener_watcher = setup_activity_feed_listener()
            log_activity("Firestore activity feed listener set up.", user="System")

        if st.session_state.firestore_activity_log:
            for entry in st.session_state.firestore_activity_log:
                timestamp_str = entry.get('timestamp', 'N/A')
                if isinstance(timestamp_str, str) and 'T' in timestamp_str:
                    try:
                        dt_object = datetime.fromisoformat(timestamp_str)
                        timestamp_str = dt_object.strftime("%Y-%m-%d %H:%M")
                    except ValueError:
                        pass
                st.markdown(f"""
                <div style="background-color: {'#3A3A3A' if st.session_state.get('dark_mode_main') else '#f0f2f6'}; padding: 10px; border-radius: 8px; margin-bottom: 5px; font-size: 0.9em; color: {'#E0E0E0' if st.session_state.get('dark_mode_main') else '#333'};">
                    **[{timestamp_str}] {entry.get('user', 'Unknown')}:** {entry.get('message', 'No message')}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recent activity to display yet.")

    with tab_pipeline:
        st.subheader("Shared Candidate Pipeline (Mock)")
        st.info("This section is a placeholder for a collaborative candidate pipeline. A full implementation would involve linking to actual candidate data and status updates, allowing team members to collectively manage and track candidates through the hiring process.")
        st.write("Imagine a dynamic table here where team members can update candidate statuses, assign ownership, and add comments in real-time.")
        
        # Mock shared pipeline data for demonstration
        mock_pipeline = [
            {"Candidate": "Jane Doe", "Status": "Interview Scheduled", "Assigned To": "Alice Johnson"},
            {"Candidate": "John Smith", "Status": "Offer Extended", "Assigned To": "Bob Williams"},
            {"Candidate": "Emily White", "Status": "Screening", "Assigned To": "Charlie Brown"},
            {"Candidate": "David Lee", "Status": "Onboarding", "Assigned To": "You"},
        ]
        pipeline_df = pd.DataFrame(mock_pipeline)
        st.dataframe(pipeline_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Pipeline Actions (Mock)")
        st.write("These actions would allow team members to interact with the pipeline:")
        col_pipe1, col_pipe2 = st.columns(2)
        with col_pipe1:
            st.button("Add Candidate to Pipeline (Mock)", key="mock_add_candidate")
        with col_pipe2:
            st.button("Update Candidate Status (Mock)", key="mock_update_status")
        st.button("Assign Candidate to Team Member (Mock)", key="mock_assign_candidate")

