import streamlit as st
import requests # For making HTTP requests to the REST API
from datetime import datetime
import json
import os
import pandas as pd # Import pandas for DataFrame

# --- Firebase REST API Configuration ---
# Global variables from Canvas environment (assuming they are set by the environment)
# These are typically set in main.py, but we'll ensure they are accessible here.
# For security, in a real app, the API_KEY might be handled server-side.
FIREBASE_PROJECT_ID = os.environ.get('__app_id', 'default-app-id')
# In a real scenario, you'd get FIREBASE_WEB_API_KEY from a secure source,
# potentially passed from main.py or an environment variable.
# For this Canvas environment, we'll assume it's available or use a placeholder.
# You might need to manually set this in your main.py or environment if not automatically provided.
FIREBASE_WEB_API_KEY = os.environ.get('FIREBASE_WEB_API_KEY', 'YOUR_FIREBASE_WEB_API_KEY') # Replace with your actual Web API Key

# Base URL for Firestore REST API
FIRESTORE_BASE_URL = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents"

# --- Helper function for converting Python dict to Firestore REST API format ---
def to_firestore_format(data: dict) -> dict:
    """Converts a Python dictionary to Firestore REST API 'fields' format."""
    fields = {}
    for key, value in data.items():
        if isinstance(value, str):
            fields[key] = {"stringValue": value}
        elif isinstance(value, int):
            fields[key] = {"integerValue": str(value)} # Firestore expects string for integerValue
        elif isinstance(value, float):
            fields[key] = {"doubleValue": value}
        elif isinstance(value, bool):
            fields[key] = {"booleanValue": value}
        elif isinstance(value, datetime):
            fields[key] = {"timestampValue": value.isoformat() + "Z"} # ISO 8601 with 'Z' for UTC
        elif isinstance(value, list):
            # For lists, convert each item and wrap in arrayValue
            array_values = []
            for item in value:
                if isinstance(item, str):
                    array_values.append({"stringValue": item})
                elif isinstance(item, int):
                    array_values.append({"integerValue": str(item)})
                elif isinstance(item, float):
                    array_values.append({"doubleValue": item})
                elif isinstance(item, bool):
                    array_values.append({"booleanValue": item})
                # Add more types as needed for list elements
            fields[key] = {"arrayValue": {"values": array_values}}
        elif isinstance(value, dict):
            # For nested dictionaries (maps), recursively convert
            fields[key] = {"mapValue": {"fields": to_firestore_format(value)}}
        elif value is None:
            fields[key] = {"nullValue": None}
        else:
            # Fallback for other types, try to stringify
            fields[key] = {"stringValue": str(value)}
    return {"fields": fields}

# --- Helper function for converting Firestore REST API format to Python dict ---
def from_firestore_format(firestore_data: dict) -> dict:
    """Converts Firestore REST API 'fields' format to a Python dictionary."""
    data = {}
    if "fields" not in firestore_data:
        return data # Or raise an error if expected
    
    for key, value_obj in firestore_data["fields"].items():
        if "stringValue" in value_obj:
            data[key] = value_obj["stringValue"]
        elif "integerValue" in value_obj:
            data[key] = int(value_obj["integerValue"])
        elif "doubleValue" in value_obj:
            data[key] = float(value_obj["doubleValue"])
        elif "booleanValue" in value_obj:
            data[key] = value_obj["booleanValue"]
        elif "timestampValue" in value_obj:
            # Remove 'Z' and parse
            try:
                data[key] = datetime.fromisoformat(value_obj["timestampValue"].replace('Z', ''))
            except ValueError:
                data[key] = value_obj["timestampValue"] # Keep as string if parsing fails
        elif "arrayValue" in value_obj and "values" in value_obj["arrayValue"]:
            # Recursively convert array elements
            data[key] = [from_firestore_format({"fields": {"_": item}})["_"] for item in value_obj["arrayValue"]["values"]]
        elif "mapValue" in value_obj and "fields" in value_obj["mapValue"]:
            # Recursively convert map values
            data[key] = from_firestore_format({"fields": value_obj["mapValue"]["fields"]})
        elif "nullValue" in value_obj:
            data[key] = None
        # Add more types as needed
    return data

# --- Helper function for activity logging ---
def log_activity(message: str, user: str = None):
    """
    Logs an activity with a timestamp to Firestore (via REST API) and session state.
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

    # Persist to Firestore via REST API
    try:
        collection_url = f"{FIRESTORE_BASE_URL}/artifacts/{app_id}/public/data/activity_feed?key={FIREBASE_WEB_API_KEY}"
        payload = to_firestore_format({
            "message": message,
            "user": user,
            "timestamp": datetime.now() # Use current datetime for logging
        })
        response = requests.post(collection_url, json=payload)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        st.error(f"Error logging activity to Firestore via REST API: {e}")
        # Note: If this error persists, check your Firebase Web API Key and network.

# --- Data Fetching Functions (replacing on_snapshot listeners) ---

def fetch_collection_data(collection_path: str, order_by_field: str = None, limit: int = 20):
    """Fetches documents from a Firestore collection via REST API."""
    url = f"{FIRESTORE_BASE_URL}/{collection_path}?key={FIREBASE_WEB_API_KEY}"
    
    # For ordering and limiting, we need to use the runQuery endpoint for structured queries
    # For simplicity, we'll fetch all and then sort/limit in Python for now.
    # A more efficient way for large datasets would be to use the structuredQuery POST endpoint.
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        documents = []
        if 'documents' in data:
            for doc_entry in data['documents']:
                doc_id = doc_entry['name'].split('/')[-1] # Extract document ID
                doc_data = from_firestore_format(doc_entry)
                doc_data['id'] = doc_id # Add document ID for updates/deletions
                documents.append(doc_data)
        
        # Sort and limit in Python (less efficient for very large datasets, but simpler for REST GET)
        if order_by_field and all(order_by_field in doc for doc in documents):
            documents.sort(key=lambda x: x.get(order_by_field), reverse=True) # Assuming descending
        
        return documents[:limit]

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from {collection_path} via REST API: {e}")
        return []

def collaboration_hub_page():
    st.markdown('<div class="dashboard-header">🤝 Collaboration Hub</div>', unsafe_allow_html=True)

    st.write("This hub is designed for seamless team collaboration. Share notes, ideas, and updates with your HR team.")

    current_username = st.session_state.get('username', 'Anonymous User')
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

        new_note_content = st.text_area("Write a new shared note:", key="new_shared_note_input")

        if st.button("Add Note", key="add_note_button"):
            if new_note_content:
                try:
                    collection_path = f"artifacts/{app_id}/public/data/shared_notes"
                    url = f"{FIRESTORE_BASE_URL}/{collection_path}?key={FIREBASE_WEB_API_KEY}"
                    payload = to_firestore_format({
                        "author": current_username,
                        "content": new_note_content,
                        "timestamp": datetime.now()
                    })
                    response = requests.post(url, json=payload)
                    response.raise_for_status()
                    st.success("Note added successfully!")
                    log_activity(f"added a shared note.", user=current_username)
                    st.session_state.notes_needs_refresh = True # Flag for refresh
                    st.rerun()
                except requests.exceptions.RequestException as e:
                    st.error(f"Error adding note: {e}")
                    log_activity(f"Error adding shared note: {e}", user=current_username)
            else:
                st.warning("Please write something before adding a note.")

        st.markdown("---")
        st.subheader("Recent Shared Notes")

        # Refresh button for notes
        if st.button("Refresh Notes", key="refresh_notes_button") or st.session_state.get('notes_needs_refresh', True):
            st.session_state.shared_notes = fetch_collection_data(
                f"artifacts/{app_id}/public/data/shared_notes",
                order_by_field="timestamp",
                limit=20
            )
            st.session_state.notes_needs_refresh = False # Reset flag

        if st.session_state.shared_notes:
            for note in st.session_state.shared_notes:
                timestamp_obj = note.get('timestamp')
                timestamp_str = timestamp_obj.strftime("%Y-%m-%d %H:%M") if isinstance(timestamp_obj, datetime) else str(timestamp_obj)

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
                    collection_path = f"artifacts/{app_id}/public/data/team_tasks"
                    url = f"{FIRESTORE_BASE_URL}/{collection_path}?key={FIREBASE_WEB_API_KEY}"
                    payload = to_firestore_format({
                        "description": new_task_description,
                        "assignee": new_task_assignee,
                        "status": "pending",
                        "created_by": current_username,
                        "created_at": datetime.now()
                    })
                    response = requests.post(url, json=payload)
                    response.raise_for_status()
                    st.success("Task added successfully!")
                    log_activity(f"added a new task: '{new_task_description}'.", user=current_username)
                    st.session_state.tasks_needs_refresh = True
                    st.rerun()
                except requests.exceptions.RequestException as e:
                    st.error(f"Error adding task: {e}")
                    log_activity(f"Error adding task: {e}", user=current_username)
            else:
                st.warning("Please provide a task description and assignee.")

        st.markdown("---")
        st.subheader("Current Tasks")

        if st.button("Refresh Tasks", key="refresh_tasks_button") or st.session_state.get('tasks_needs_refresh', True):
            st.session_state.team_tasks = fetch_collection_data(
                f"artifacts/{app_id}/public/data/team_tasks",
                order_by_field="created_at",
                limit=20
            )
            st.session_state.tasks_needs_refresh = False

        if st.session_state.team_tasks:
            for task in st.session_state.team_tasks:
                timestamp_obj = task.get('created_at')
                timestamp_str = timestamp_obj.strftime("%Y-%m-%d %H:%M") if isinstance(timestamp_obj, datetime) else str(timestamp_obj)

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
                if task.get('status') == 'pending':
                    if st.button(f"Mark as Complete", key=f"complete_task_{task['id']}"):
                        try:
                            doc_path = f"artifacts/{app_id}/public/data/team_tasks/{task['id']}"
                            url = f"{FIRESTORE_BASE_URL}/{doc_path}?key={FIREBASE_WEB_API_KEY}&updateMask.fieldPaths=status"
                            payload = to_firestore_format({"status": "completed"})
                            response = requests.patch(url, json=payload) # Use PATCH for partial update
                            response.raise_for_status()
                            st.success("Task marked as complete!")
                            log_activity(f"marked task '{task['description']}' as complete.", user=current_username)
                            st.session_state.tasks_needs_refresh = True
                            st.rerun()
                        except requests.exceptions.RequestException as e:
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
                    collection_path = f"artifacts/{app_id}/public/data/team_announcements"
                    url = f"{FIRESTORE_BASE_URL}/{collection_path}?key={FIREBASE_WEB_API_KEY}"
                    payload = to_firestore_format({
                        "title": new_announcement_title,
                        "content": new_announcement_content,
                        "author": current_username,
                        "timestamp": datetime.now()
                    })
                    response = requests.post(url, json=payload)
                    response.raise_for_status()
                    st.success("Announcement posted successfully!")
                    log_activity(f"posted an announcement: '{new_announcement_title}'.", user=current_username)
                    st.session_state.announcements_needs_refresh = True
                    st.rerun()
                except requests.exceptions.RequestException as e:
                    st.error(f"Error posting announcement: {e}")
                    log_activity(f"Error posting announcement: {e}", user=current_username)
            else:
                st.warning("Please provide both a title and content for the announcement.")

        st.markdown("---")
        st.subheader("Recent Announcements")

        if st.button("Refresh Announcements", key="refresh_announcements_button") or st.session_state.get('announcements_needs_refresh', True):
            st.session_state.team_announcements = fetch_collection_data(
                f"artifacts/{app_id}/public/data/team_announcements",
                order_by_field="timestamp",
                limit=10
            )
            st.session_state.announcements_needs_refresh = False

        if st.session_state.team_announcements:
            for announcement in st.session_state.team_announcements:
                timestamp_obj = announcement.get('timestamp')
                timestamp_str = timestamp_obj.strftime("%Y-%m-%d %H:%M") if isinstance(timestamp_obj, datetime) else str(timestamp_obj)

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
                    if "@" not in new_member_email or "." not in new_member_email:
                        st.error("Please enter a valid email address for the new member.")
                    else:
                        try:
                            # Use the email as the document ID for easy lookup and uniqueness
                            doc_path = f"artifacts/{app_id}/public/data/team_members/{new_member_email}"
                            url = f"{FIRESTORE_BASE_URL}/{doc_path}?key={FIREBASE_WEB_API_KEY}"
                            
                            # Check if member exists (GET request)
                            check_response = requests.get(url)
                            if check_response.status_code == 200:
                                st.warning(f"A team member with email '{new_member_email}' already exists.")
                            else:
                                payload = to_firestore_format({
                                    "name": new_member_name,
                                    "email": new_member_email,
                                    "role": new_member_role,
                                    "status": "active", # Default status
                                    "added_by": current_username,
                                    "added_at": datetime.now()
                                })
                                # Use PATCH to create or update document by ID
                                response = requests.patch(url, json=payload)
                                response.raise_for_status()
                                st.success(f"Team member '{new_member_name}' added successfully!")
                                log_activity(f"added new team member: '{new_member_name}' ({new_member_email}).", user=current_username)
                                st.session_state.members_needs_refresh = True
                                st.rerun()
                        except requests.exceptions.RequestException as e:
                            st.error(f"Error adding team member: {e}")
                            log_activity(f"Error adding team member: {e}", user=current_username)
                else:
                    st.warning("Please fill in all fields for the new team member.")

        st.markdown("---")
        st.subheader("Current Team Members")

        if st.button("Refresh Team Members", key="refresh_members_button") or st.session_state.get('members_needs_refresh', True):
            st.session_state.team_members = fetch_collection_data(
                f"artifacts/{app_id}/public/data/team_members",
                order_by_field="added_at",
                limit=50
            )
            st.session_state.members_needs_refresh = False

        if st.session_state.team_members:
            team_members_df = pd.DataFrame(st.session_state.team_members)
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
            recipient_options = [member['email'] for member in st.session_state.team_members if member['email'] != current_username]
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
                        collection_path = f"artifacts/{app_id}/public/data/team_events"
                        url = f"{FIRESTORE_BASE_URL}/{collection_path}?key={FIREBASE_WEB_API_KEY}"
                        payload = to_firestore_format({
                            "title": event_title,
                            "date": str(event_date), # Store date as string (YYYY-MM-DD)
                            "time": event_time,
                            "description": event_description,
                            "created_by": current_username,
                            "created_at": datetime.now()
                        })
                        response = requests.post(url, json=payload)
                        response.raise_for_status()
                        st.success("Event added successfully!")
                        log_activity(f"added a new calendar event: '{event_title}'.", user=current_username)
                        st.session_state.events_needs_refresh = True
                        st.rerun()
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error adding event: {e}")
                        log_activity(f"Error adding event: {e}", user=current_username)
                else:
                    st.warning("Please fill in event title, date, and time.")

        st.markdown("---")
        st.subheader("Upcoming Events")

        if st.button("Refresh Events", key="refresh_events_button") or st.session_state.get('events_needs_refresh', True):
            st.session_state.team_events = fetch_collection_data(
                f"artifacts/{app_id}/public/data/team_events",
                order_by_field="date", # Order by date
                limit=10
            )
            st.session_state.events_needs_refresh = False

        if st.session_state.team_events:
            # Filter out past events for "Upcoming Events" display
            today = datetime.now().date()
            upcoming_events = []
            for event in st.session_state.team_events:
                event_date_str = event.get('date', '1900-01-01')
                try:
                    event_date_obj = datetime.strptime(event_date_str, '%Y-%m-%d').date()
                    if event_date_obj >= today:
                        upcoming_events.append(event)
                except ValueError:
                    pass # Ignore malformed dates

            # Sort by time for events on the same day
            upcoming_events.sort(key=lambda x: (datetime.strptime(x['date'], '%Y-%m-%d').date(), x.get('time', '00:00')))

            if upcoming_events:
                for event in upcoming_events:
                    timestamp_obj = event.get('created_at')
                    created_at_str = timestamp_obj.strftime("%Y-%m-%d %H:%M") if isinstance(timestamp_obj, datetime) else str(timestamp_obj)

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
                            Added by: {event.get('created_by', 'Unknown')} at {created_at_str}
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
                        collection_path = f"artifacts/{app_id}/public/data/resource_library"
                        url = f"{FIRESTORE_BASE_URL}/{collection_path}?key={FIREBASE_WEB_API_KEY}"
                        payload = to_firestore_format({
                            "name": resource_name,
                            "url": resource_url,
                            "description": resource_description,
                            "category": resource_category,
                            "uploaded_by": current_username,
                            "uploaded_at": datetime.now()
                        })
                        response = requests.post(url, json=payload)
                        response.raise_for_status()
                        st.success("Resource added successfully!")
                        log_activity(f"added a new resource: '{resource_name}'.", user=current_username)
                        st.session_state.resources_needs_refresh = True
                        st.rerun()
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error adding resource: {e}")
                        log_activity(f"Error adding resource: {e}", user=current_username)
                else:
                    st.warning("Please provide a resource name.")

        st.markdown("---")
        st.subheader("Available Resources")

        if st.button("Refresh Resources", key="refresh_resources_button") or st.session_state.get('resources_needs_refresh', True):
            st.session_state.resource_library = fetch_collection_data(
                f"artifacts/{app_id}/public/data/resource_library",
                order_by_field="uploaded_at",
                limit=20
            )
            st.session_state.resources_needs_refresh = False

        if st.session_state.resource_library:
            for resource in st.session_state.resource_library:
                timestamp_obj = resource.get('uploaded_at')
                timestamp_str = timestamp_obj.strftime("%Y-%m-%d %H:%M") if isinstance(timestamp_obj, datetime) else str(timestamp_obj)

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
                            collection_path = f"artifacts/{app_id}/public/data/team_polls"
                            url = f"{FIRESTORE_BASE_URL}/{collection_path}?key={FIREBASE_WEB_API_KEY}"
                            payload = to_firestore_format({
                                "question": poll_question,
                                "options": options,
                                "votes": {option: 0 for option in options}, # Initialize votes to 0
                                "active": True,
                                "created_by": current_username,
                                "created_at": datetime.now()
                            })
                            response = requests.post(url, json=payload)
                            response.raise_for_status()
                            st.success("Poll created successfully!")
                            log_activity(f"created a new poll: '{poll_question}'.", user=current_username)
                            st.session_state.polls_needs_refresh = True
                            st.rerun()
                        except requests.exceptions.RequestException as e:
                            st.error(f"Error creating poll: {e}")
                            log_activity(f"Error creating poll: {e}", user=current_username)
                else:
                    st.warning("Please provide a poll question and options.")

        st.markdown("---")
        st.subheader("Active Polls")

        if st.button("Refresh Polls", key="refresh_polls_button") or st.session_state.get('polls_needs_refresh', True):
            # For active polls, we need to filter. This requires a structured query POST request.
            # For simplicity with GET, we'll fetch all and filter in Python.
            all_polls = fetch_collection_data(
                f"artifacts/{app_id}/public/data/team_polls",
                order_by_field="created_at",
                limit=5 # Limit for active polls
            )
            st.session_state.active_polls = [poll for poll in all_polls if poll.get('active') == True]
            st.session_state.polls_needs_refresh = False

        if st.session_state.active_polls:
            for poll in st.session_state.active_polls:
                timestamp_obj = poll.get('created_at')
                timestamp_str = timestamp_obj.strftime("%Y-%m-%d %H:%M") if isinstance(timestamp_obj, datetime) else str(timestamp_obj)

                st.markdown(f"""
                <div style="background-color: {'#3A3A3A' if st.session_state.get('dark_mode_main') else '#f0f2f6'}; padding: 15px; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <h4 style="color: {'#00cec9' if st.session_state.get('dark_mode_main') else '#00cec9'}; margin-bottom: 5px;">{poll.get('question', 'No Question')}</h4>
                    <p style="font-size: 0.9em; color: {'#BBBBBB' if st.session_state.get('dark_mode_main') else '#666'}; margin-bottom: 5px;">
                        Created by: **{poll.get('created_by', 'Unknown')}** at {timestamp_str}
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
                                doc_path = f"artifacts/{app_id}/public/data/team_polls/{poll['id']}"
                                url = f"{FIRESTORE_BASE_URL}/{doc_path}?key={FIREBASE_WEB_API_KEY}&updateMask.fieldPaths=votes"
                                
                                current_votes = poll.get('votes', {})
                                current_votes[selected_option] = current_votes.get(selected_option, 0) + 1
                                
                                # Only update the 'votes' field
                                payload = {"fields": {"votes": {"mapValue": {"fields": {
                                    k: {"integerValue": str(v)} for k, v in current_votes.items()
                                }}}}}
                                
                                response = requests.patch(url, json=payload)
                                response.raise_for_status()
                                st.session_state[user_voted_key] = True # Mark user as voted in session
                                st.success("Vote submitted successfully!")
                                log_activity(f"voted on poll '{poll['question']}' for option '{selected_option}'.", user=current_username)
                                st.session_state.polls_needs_refresh = True
                                st.rerun() # Rerun to update the displayed results
                            except requests.exceptions.RequestException as e:
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
                admin_usernames = ("admin@forscreenerpro", "admin@forscreenerpro2", "manav.nagpal2005@gmail.com")
                if current_username == poll.get('created_by') or current_username in admin_usernames:
                    if poll.get('active') and st.button("Close Poll", key=f"close_poll_{poll['id']}"):
                        try:
                            doc_path = f"artifacts/{app_id}/public/data/team_polls/{poll['id']}"
                            url = f"{FIRESTORE_BASE_URL}/{doc_path}?key={FIREBASE_WEB_API_KEY}&updateMask.fieldPaths=active"
                            payload = to_firestore_format({"active": False})
                            response = requests.patch(url, json=payload)
                            response.raise_for_status()
                            st.success("Poll closed successfully!")
                            log_activity(f"closed poll: '{poll['question']}'.", user=current_username)
                            st.session_state.polls_needs_refresh = True
                            st.rerun()
                        except requests.exceptions.RequestException as e:
                            st.error(f"Error closing poll: {e}")
                            log_activity(f"Error closing poll: {e}", user=current_username)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No active polls. Be the first to create one!")

    with tab_activity:
        st.subheader("Recent Activity Feed")
        st.write("See a chronological log of recent actions within the Collaboration Hub across all users.")

        if st.button("Refresh Activity Feed", key="refresh_activity_button") or st.session_state.get('activity_needs_refresh', True):
            st.session_state.firestore_activity_log = fetch_collection_data(
                f"artifacts/{app_id}/public/data/activity_feed",
                order_by_field="timestamp",
                limit=30
            )
            st.session_state.activity_needs_refresh = False

        if st.session_state.firestore_activity_log:
            for entry in st.session_state.firestore_activity_log:
                timestamp_obj = entry.get('timestamp')
                timestamp_str = timestamp_obj.strftime("%Y-%m-%d %H:%M") if isinstance(timestamp_obj, datetime) else str(timestamp_obj)

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

