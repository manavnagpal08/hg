import streamlit as st
import requests # For making HTTP requests to the REST API
from datetime import datetime
import json
import os
import pandas as pd # Import pandas for DataFrame
import collections # For defaultdict

# --- Firebase REST API Configuration ---
# These will be passed from main.py
# FIREBASE_WEB_API_KEY
# FIRESTORE_BASE_URL
# app_id

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
                elif isinstance(item, dict): # Handle nested dicts in lists
                    array_values.append({"mapValue": {"fields": to_firestore_format(item)['fields']}})
                else: # Fallback for other types in list
                    array_values.append({"stringValue": str(item)})
            fields[key] = {"arrayValue": {"values": array_values}}
        elif isinstance(value, dict):
            # For nested dictionaries (maps), recursively convert
            fields[key] = {"mapValue": {"fields": to_firestore_format(value)['fields']}}
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
            data[key] = [from_firestore_format({"fields": {"_": item}})["_"] if "mapValue" not in item else from_firestore_format({"fields": item["mapValue"]["fields"]}) for item in value_obj["arrayValue"]["values"]]
        elif "mapValue" in value_obj and "fields" in value_obj["mapValue"]:
            # Recursively convert map values
            data[key] = from_firestore_format({"fields": value_obj["mapValue"]["fields"]})
        elif "nullValue" in value_obj:
            data[key] = None
        # Add more types as needed
    return data

# --- Helper function for activity logging ---
# This function now takes FIREBASE_WEB_API_KEY and FIRESTORE_BASE_URL as arguments
def log_activity(message: str, user: str, FIREBASE_WEB_API_KEY: str, FIRESTORE_BASE_URL: str, app_id: str):
    """
    Logs an activity with a timestamp to Firestore (via REST API) and session state.
    This log is public/common across all companies.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {user}: {message}"

    # Add to session state log (for immediate display in the Activity Feed tab)
    if 'activity_log' not in st.session_state:
        st.session_state.activity_log = []
    st.session_state.activity_log.insert(0, log_entry) # Add to the beginning for most recent first
    st.session_state.activity_log = st.session_state.activity_log[:50] # Keep log size manageable

    # Persist to Firestore via REST API
    try:
        # This collection is intentionally public for general app activity feed
        collection_url = f"{FIRESTORE_BASE_URL}/artifacts/{app_id}/public/data/activity_feed"
        payload = to_firestore_format({
            "message": message,
            "user": user,
            "timestamp": datetime.now() # Use current datetime for logging
        })
        response = requests.post(collection_url, json=payload)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        # Log to console, but don't show error to user for background logging
        print(f"Error logging activity to Firestore via REST API: {e}")

# --- Data Fetching Functions (replacing on_snapshot listeners) ---
# These functions now take FIREBASE_WEB_API_KEY and FIRESTORE_BASE_URL as arguments
def fetch_collection_data(collection_path: str, FIREBASE_WEB_API_KEY: str, FIRESTORE_BASE_URL: str, order_by_field: str = None, limit: int = 20):
    """Fetches documents from a Firestore collection via REST API."""
    url = f"{FIRESTORE_BASE_URL}/{collection_path}?key={FIREBASE_WEB_API_KEY}"
    
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
        # For data fetching, it's okay to show an error to the user
        st.error(f"Error fetching data from {collection_path} via REST API: {e}")
        return []

def save_document_to_firestore(collection_path, doc_id, data, api_key, base_url):
    """Saves a document to Firestore using PATCH (create or update)."""
    url = f"{base_url}/{collection_path}/{doc_id}?key={api_key}"
    firestore_data = to_firestore_format(data)
    try:
        res = requests.patch(url, json=firestore_data)
        res.raise_for_status() # Raise an exception for HTTP errors
        return True, res.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Firestore save error: {e}")
        return False, str(e)

def add_document_to_firestore_collection(collection_path, data, api_key, base_url):
    """Adds a new document to a Firestore collection (Firestore assigns ID)."""
    url = f"{base_url}/{collection_path}?key={api_key}"
    firestore_data = to_firestore_format(data)
    try:
        res = requests.post(url, json=firestore_data)
        res.raise_for_status()
        return True, res.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Firestore add error: {e}")
        return False, str(e)

# --- Main Collaboration Hub Page Function ---
# Now accepts app_id, FIREBASE_WEB_API_KEY, and FIRESTORE_BASE_URL as parameters
def collaboration_hub_page(app_id: str, FIREBASE_WEB_API_KEY: str, FIRESTORE_BASE_URL: str):
    st.markdown('<div class="dashboard-header">🤝 Collaboration Hub</div>', unsafe_allow_html=True)

    st.write("This hub is designed for seamless team collaboration. Share notes, ideas, and updates with your HR team.")

    current_username = st.session_state.get('username', 'Anonymous User')
    current_user_id = st.session_state.get('user_id', current_username)
    # Get the company name from session state and sanitize it for Firestore path
    user_company = st.session_state.get('user_company', 'default_company').replace(' ', '_').lower()

    st.info(f"You are logged in as: **{current_username}** (Company: **{st.session_state.get('user_company', 'N/A')}**)")
    # Temporary debug message for user_company
    st.write(f"**DEBUG: Current Company for Data Isolation:** `{user_company}`")
    st.markdown("---")

    # --- Tabs for different collaboration features ---
    tab_notes, tab_tasks, tab_announcements, tab_members, tab_messages, tab_files, tab_calendar, tab_resources, tab_polls, tab_activity, tab_pipeline = st.tabs([
        "📝 Shared Notes", "✅ Team Tasks", "📢 Announcements", "👥 Team Members", "💬 Direct Messages", "📂 File Sharing",
        "🗓️ Team Calendar", "📚 Resource Library", "📊 Quick Polls", "⚡ Activity Feed", "📋 Candidate Pipeline"
    ])

    with tab_notes:
        st.subheader("Shared Team Notes")
        st.info(f"Notes shared here are visible only to HRs from **{st.session_state.get('user_company', 'your company')}**.")

        new_note_content = st.text_area("Write a new shared note:", key="new_shared_note_input")

        if st.button("Add Note", key="add_note_button"):
            if new_note_content:
                try:
                    # Company-specific path
                    collection_path = f"artifacts/{app_id}/companies/{user_company}/shared_notes"
                    success, response = add_document_to_firestore_collection(
                        collection_path,
                        {
                            "author": current_username,
                            "content": new_note_content,
                            "timestamp": datetime.now()
                        },
                        FIREBASE_WEB_API_KEY,
                        FIRESTORE_BASE_URL
                    )
                    if success:
                        st.success("Note added successfully!")
                        log_activity(f"added a shared note to {st.session_state.get('user_company', 'their company')}'s notes.", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                        st.session_state.notes_needs_refresh = True # Flag for refresh
                        st.rerun()
                    else:
                        st.error(f"Error adding note: {response}")
                        log_activity(f"Error adding shared note: {response}", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
            else:
                st.warning("Please write something before adding a note.")

        st.markdown("---")
        st.subheader("Recent Shared Notes")

        # Refresh button for notes
        if st.button("Refresh Notes", key="refresh_notes_button") or st.session_state.get('notes_needs_refresh', True):
            st.session_state.shared_notes = fetch_collection_data(
                f"artifacts/{app_id}/companies/{user_company}/shared_notes", # Company-specific path
                FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY,
                FIRESTORE_BASE_URL=FIRESTORE_BASE_URL,
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
        st.info(f"Tasks managed here are visible only to HRs from **{st.session_state.get('user_company', 'your company')}**.")
        st.write("Assign and track tasks for your team.")

        new_task_description = st.text_input("New Task Description:", key="new_task_input")
        new_task_assignee = st.text_input("Assignee (e.g., 'Alice', 'Bob', 'All'):", help="Enter the name or email of the assignee.", key="new_task_assignee")

        if st.button("Add Task", key="add_task_button"):
            if new_task_description and new_task_assignee:
                try:
                    # Company-specific path
                    collection_path = f"artifacts/{app_id}/companies/{user_company}/team_tasks"
                    success, response = add_document_to_firestore_collection(
                        collection_path,
                        {
                            "description": new_task_description,
                            "assignee": new_task_assignee,
                            "status": "pending",
                            "created_by": current_username,
                            "created_at": datetime.now()
                        },
                        FIREBASE_WEB_API_KEY,
                        FIRESTORE_BASE_URL
                    )
                    if success:
                        st.success("Task added successfully!")
                        log_activity(f"added a new task: '{new_task_description}'.", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                        st.session_state.tasks_needs_refresh = True
                        st.rerun()
                    else:
                        st.error(f"Error adding task: {response}")
                        log_activity(f"Error adding task: {response}", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
            else:
                st.warning("Please provide a task description and assignee.")

        st.markdown("---")
        st.subheader("Current Tasks")

        if st.button("Refresh Tasks", key="refresh_tasks_button") or st.session_state.get('tasks_needs_refresh', True):
            st.session_state.team_tasks = fetch_collection_data(
                f"artifacts/{app_id}/companies/{user_company}/team_tasks", # Company-specific path
                FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY,
                FIRESTORE_BASE_URL=FIRESTORE_BASE_URL,
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
                            doc_path = f"artifacts/{app_id}/companies/{user_company}/team_tasks/{task['id']}" # Company-specific path
                            success, response = save_document_to_firestore(
                                doc_path,
                                task['id'],
                                {"status": "completed"}, # Only update status
                                FIREBASE_WEB_API_KEY,
                                FIRESTORE_BASE_URL
                            )
                            if success:
                                st.success("Task marked as complete!")
                                log_activity(f"marked task '{task['description']}' as complete.", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                                st.session_state.tasks_needs_refresh = True
                                st.rerun()
                            else:
                                st.error(f"Error updating task: {response}")
                                log_activity(f"Error updating task '{task['description']}': {response}", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                        except Exception as e:
                            st.error(f"An unexpected error occurred: {e}")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No tasks added yet. Add a new task above!")

    with tab_announcements:
        st.subheader("Team Announcements")
        st.info(f"Announcements posted here are visible only to HRs from **{st.session_state.get('user_company', 'your company')}**.")
        st.write("Post important announcements for the entire team.")

        new_announcement_title = st.text_input("Announcement Title:", key="new_announcement_title")
        new_announcement_content = st.text_area("Announcement Details:", key="new_announcement_content")

        if st.button("Post Announcement", key="post_announcement_button"):
            if new_announcement_title and new_announcement_content:
                try:
                    # Company-specific path
                    collection_path = f"artifacts/{app_id}/companies/{user_company}/team_announcements"
                    success, response = add_document_to_firestore_collection(
                        collection_path,
                        {
                            "title": new_announcement_title,
                            "content": new_announcement_content,
                            "author": current_username,
                            "timestamp": datetime.now()
                        },
                        FIREBASE_WEB_API_KEY,
                        FIRESTORE_BASE_URL
                    )
                    if success:
                        st.success("Announcement posted successfully!")
                        log_activity(f"posted an announcement: '{new_announcement_title}'.", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                        st.session_state.announcements_needs_refresh = True
                        st.rerun()
                    else:
                        st.error(f"Error posting announcement: {response}")
                        log_activity(f"Error posting announcement: {response}", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
            else:
                st.warning("Please provide both a title and content for the announcement.")

        st.markdown("---")
        st.subheader("Recent Announcements")

        if st.button("Refresh Announcements", key="refresh_announcements_button") or st.session_state.get('announcements_needs_refresh', True):
            st.session_state.team_announcements = fetch_collection_data(
                f"artifacts/{app_id}/companies/{user_company}/team_announcements", # Company-specific path
                FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY,
                FIRESTORE_BASE_URL=FIRESTORE_BASE_URL,
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
        st.info(f"Team members listed here belong to **{st.session_state.get('user_company', 'your company')}**.")
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
                            # Company-specific path
                            doc_path = f"artifacts/{app_id}/companies/{user_company}/team_members/{new_member_email.replace('.', '_').replace('@', '_')}"
                            url = f"{FIRESTORE_BASE_URL}/{doc_path}?key={FIREBASE_WEB_API_KEY}"
                            
                            # Check if member exists (GET request)
                            check_response = requests.get(url)
                            if check_response.status_code == 200:
                                st.warning(f"A team member with email '{new_member_email}' already exists in this company.")
                            else:
                                payload = {
                                    "name": new_member_name,
                                    "email": new_member_email,
                                    "role": new_member_role,
                                    "status": "active", # Default status
                                    "added_by": current_username,
                                    "added_at": datetime.now()
                                }
                                # Use PATCH to create or update document by ID
                                success, response = save_document_to_firestore(
                                    f"artifacts/{app_id}/companies/{user_company}/team_members", # Company-specific path
                                    new_member_email.replace('.', '_').replace('@', '_'), # Document ID
                                    payload,
                                    FIREBASE_WEB_API_KEY,
                                    FIRESTORE_BASE_URL
                                )
                                if success:
                                    st.success(f"Team member '{new_member_name}' added successfully!")
                                    log_activity(f"added new team member: '{new_member_name}' ({new_member_email}) to {st.session_state.get('user_company', 'their company')}'s team.", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                                    st.session_state.members_needs_refresh = True
                                    st.rerun()
                                else:
                                    st.error(f"Error adding team member: {response}")
                                    log_activity(f"Error adding team member: {response}", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                        except Exception as e:
                            st.error(f"An unexpected error occurred: {e}")
                else:
                    st.warning("Please fill in all fields for the new team member.")

        st.markdown("---")
        st.subheader("Current Team Members")

        if st.button("Refresh Team Members", key="refresh_members_button") or st.session_state.get('members_needs_refresh', True):
            st.session_state.team_members = fetch_collection_data(
                f"artifacts/{app_id}/companies/{user_company}/team_members", # Company-specific path
                FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY,
                FIRESTORE_BASE_URL=FIRESTORE_BASE_URL,
                order_by_field="added_at",
                limit=50
            )
            st.session_state.members_needs_refresh = False

        if st.session_state.team_members:
            team_members_df = pd.DataFrame(st.session_state.team_members)
            display_cols = ['name', 'email', 'role', 'status', 'added_by']
            st.dataframe(team_members_df[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No team members added yet for this company. Add new members using the form above!")


    with tab_messages:
        st.subheader("Direct Messages (Simulated Persistence)")
        st.info(f"Messages sent here are stored for **{st.session_state.get('user_company', 'your company')}** members. This is a simulated chat; real-time updates are not supported.")
        
        # Ensure direct_messages are loaded for the current company
        if 'direct_messages' not in st.session_state or st.session_state.get('messages_needs_refresh', True):
            st.session_state.direct_messages = fetch_collection_data(
                f"artifacts/{app_id}/companies/{user_company}/direct_messages",
                FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY,
                FIRESTORE_BASE_URL=FIRESTORE_BASE_URL,
                order_by_field="timestamp",
                limit=50 # Limit messages for performance
            )
            st.session_state.messages_needs_refresh = False

        # Populate recipient list from actual team members if available
        recipient_options = [member['email'] for member in st.session_state.get('team_members', []) if member['email'] != current_username]
        if not recipient_options:
            recipient_options = ["No other members available"]
        
        selected_recipient = st.selectbox("Select Recipient", recipient_options, key="dm_recipient_select")
        message_content = st.text_area(f"Message to {selected_recipient}:", key="dm_message_input")

        if st.button("Send Message", key="send_dm_button"):
            if message_content and selected_recipient != "No other members available":
                try:
                    collection_path = f"artifacts/{app_id}/companies/{user_company}/direct_messages"
                    success, response = add_document_to_firestore_collection(
                        collection_path,
                        {
                            "sender": current_username,
                            "recipient": selected_recipient,
                            "content": message_content,
                            "timestamp": datetime.now()
                        },
                        FIREBASE_WEB_API_KEY,
                        FIRESTORE_BASE_URL
                    )
                    if success:
                        st.success(f"Message sent to {selected_recipient}!")
                        log_activity(f"sent a message to '{selected_recipient}'.", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                        st.session_state.messages_needs_refresh = True
                        st.rerun()
                    else:
                        st.error(f"Error sending message: {response}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
            else:
                st.warning("Please type a message and select a valid recipient.")

        st.markdown("---")
        st.subheader("Your Message History")

        if st.button("Refresh Message History", key="refresh_messages_button"):
            st.session_state.messages_needs_refresh = True
            st.rerun()

        if st.session_state.direct_messages:
            # Filter messages relevant to the current user (sent by or received by)
            user_messages = [
                msg for msg in st.session_state.direct_messages
                if msg.get('sender') == current_username or msg.get('recipient') == current_username
            ]
            # Sort by timestamp (most recent first)
            user_messages.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)

            if user_messages:
                for msg in user_messages:
                    timestamp_obj = msg.get('timestamp')
                    timestamp_str = timestamp_obj.strftime("%Y-%m-%d %H:%M") if isinstance(timestamp_obj, datetime) else str(timestamp_obj)
                    
                    sender = msg.get('sender', 'Unknown')
                    recipient = msg.get('recipient', 'Unknown')
                    content = msg.get('content', 'No content')

                    if sender == current_username:
                        display_text = f"**You** to **{recipient}** at {timestamp_str}: {content}"
                        bg_color = '#00cec9' if st.session_state.get('dark_mode_main') else '#e0f7fa'
                        text_color = 'white' if st.session_state.get('dark_mode_main') else '#333'
                    else: # Message received
                        display_text = f"**{sender}** to **You** at {timestamp_str}: {content}"
                        bg_color = '#6c757d' if st.session_state.get('dark_mode_main') else '#f8f9fa'
                        text_color = 'white' if st.session_state.get('dark_mode_main') else '#333'

                    st.markdown(f"""
                    <div style="background-color: {bg_color}; padding: 10px; border-radius: 8px; margin-bottom: 5px; font-size: 0.9em; color: {text_color};">
                        {display_text}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No message history found for you in this company.")
        else:
            st.info("No messages have been sent or received yet in this company.")

    with tab_files:
        st.subheader("Shared Files (Metadata Storage Only)")
        st.info(f"""
            This section allows you to manage *metadata* about shared files (like name, description, and URL).
            Data is isolated for **{st.session_state.get('user_company', 'your company')}**.
            **Actual file storage (upload/download) is NOT supported here.**
            For real file storage, you would need to integrate with a dedicated cloud storage service (e.g., Firebase Storage, AWS S3, Google Cloud Storage) which handles binary data.
        """)
        st.write("You can list external file links and descriptions for your team.")

        with st.form("add_file_metadata_form", clear_on_submit=True):
            file_name = st.text_input("File Name/Title:", key="file_metadata_name")
            file_url = st.text_input("Direct Link to File (e.g., Google Drive, Dropbox URL):", key="file_metadata_url")
            file_description = st.text_area("Description:", key="file_metadata_description")
            add_file_metadata_button = st.form_submit_button("Add File Metadata")

            if add_file_metadata_button:
                if file_name and file_url:
                    try:
                        collection_path = f"artifacts/{app_id}/companies/{user_company}/file_metadata"
                        success, response = add_document_to_firestore_collection(
                            collection_path,
                            {
                                "name": file_name,
                                "url": file_url,
                                "description": file_description,
                                "uploaded_by": current_username,
                                "uploaded_at": datetime.now()
                            },
                            FIREBASE_WEB_API_KEY,
                            FIRESTORE_BASE_URL
                        )
                        if success:
                            st.success("File metadata added successfully!")
                            log_activity(f"added file metadata for '{file_name}'.", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                            st.session_state.files_needs_refresh = True
                            st.rerun()
                        else:
                            st.error(f"Error adding file metadata: {response}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
                else:
                    st.warning("Please provide file name and URL.")

        st.markdown("---")
        st.subheader("Shared File List")

        if st.button("Refresh File List", key="refresh_files_button") or st.session_state.get('files_needs_refresh', True):
            st.session_state.shared_files_metadata = fetch_collection_data(
                f"artifacts/{app_id}/companies/{user_company}/file_metadata",
                FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY,
                FIRESTORE_BASE_URL=FIRESTORE_BASE_URL,
                order_by_field="uploaded_at",
                limit=20
            )
            st.session_state.files_needs_refresh = False

        if st.session_state.shared_files_metadata:
            for file_meta in st.session_state.shared_files_metadata:
                timestamp_obj = file_meta.get('uploaded_at')
                timestamp_str = timestamp_obj.strftime("%Y-%m-%d %H:%M") if isinstance(timestamp_obj, datetime) else str(timestamp_obj)
                
                st.markdown(f"""
                <div style="background-color: {'#3A3A3A' if st.session_state.get('dark_mode_main') else '#f0f2f6'}; padding: 15px; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <h4 style="color: {'#00cec9' if st.session_state.get('dark_mode_main') else '#00cec9'}; margin-bottom: 5px;">
                        {file_meta.get('name', 'No Name')}
                    </h4>
                    <p style="font-size: 1.0em; color: {'#E0E0E0' if st.session_state.get('dark_mode_main') else '#333'};">
                        {file_meta.get('description', 'No description')}
                    </p>
                    {"<p><a href='" + file_meta['url'] + "' target='_blank' style='color: #00cec9;'>Open File Link</a></p>" if file_meta.get('url') else ""}
                    <p style="font-size: 0.8em; color: {'#999999' if st.session_state.get('dark_mode_main') else '#888'}; margin-top: 5px;">
                        Added by: {file_meta.get('uploaded_by', 'Unknown')} at {timestamp_str}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No file metadata added yet for this company.")

    with tab_calendar:
        st.subheader("Team Calendar")
        st.info(f"Events scheduled here are visible only to HRs from **{st.session_state.get('user_company', 'your company')}**.")
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
                        # Company-specific path
                        collection_path = f"artifacts/{app_id}/companies/{user_company}/team_events"
                        success, response = add_document_to_firestore_collection(
                            collection_path,
                            {
                                "title": event_title,
                                "date": str(event_date), # Store date as string (YYYY-MM-DD)
                                "time": event_time,
                                "description": event_description,
                                "created_by": current_username,
                                "created_at": datetime.now()
                            },
                            FIREBASE_WEB_API_KEY,
                            FIRESTORE_BASE_URL
                        )
                        if success:
                            st.success("Event added successfully!")
                            log_activity(f"added a new calendar event: '{event_title}'.", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                            st.session_state.events_needs_refresh = True
                            st.rerun()
                        else:
                            st.error(f"Error adding event: {response}")
                            log_activity(f"Error adding event: {response}", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
                else:
                    st.warning("Please fill in event title, date, and time.")

        st.markdown("---")
        st.subheader("Upcoming Events")

        if st.button("Refresh Events", key="refresh_events_button") or st.session_state.get('events_needs_refresh', True):
            st.session_state.team_events = fetch_collection_data(
                f"artifacts/{app_id}/companies/{user_company}/team_events", # Company-specific path
                FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY,
                FIRESTORE_BASE_URL=FIRESTORE_BASE_URL,
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
                st.info("No upcoming events scheduled for this company.")
        else:
            st.info("No events added yet for this company. Add a new event above!")

    with tab_resources:
        st.subheader("Resource Library")
        st.info(f"Resources added here are visible only to HRs from **{st.session_state.get('user_company', 'your company')}**.")
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
                        # Company-specific path
                        collection_path = f"artifacts/{app_id}/companies/{user_company}/resource_library"
                        success, response = add_document_to_firestore_collection(
                            collection_path,
                            {
                                "name": resource_name,
                                "url": resource_url,
                                "description": resource_description,
                                "category": resource_category,
                                "uploaded_by": current_username,
                                "uploaded_at": datetime.now()
                            },
                            FIREBASE_WEB_API_KEY,
                            FIRESTORE_BASE_URL
                        )
                        if success:
                            st.success("Resource added successfully!")
                            log_activity(f"added a new resource: '{resource_name}'.", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                            st.session_state.resources_needs_refresh = True
                            st.rerun()
                        else:
                            st.error(f"Error adding resource: {response}")
                            log_activity(f"Error adding resource: {response}", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
                else:
                    st.warning("Please provide a resource name.")

        st.markdown("---")
        st.subheader("Available Resources")

        if st.button("Refresh Resources", key="refresh_resources_button") or st.session_state.get('resources_needs_refresh', True):
            st.session_state.resource_library = fetch_collection_data(
                f"artifacts/{app_id}/companies/{user_company}/resource_library", # Company-specific path
                FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY,
                FIRESTORE_BASE_URL=FIRESTORE_BASE_URL,
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
            st.info("No resources added yet for this company. Add a new resource above!")

    with tab_polls:
        st.subheader("Quick Polls/Surveys")
        st.info(f"Polls created here are visible only to HRs from **{st.session_state.get('user_company', 'your company')}**.")
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
                            # Company-specific path
                            collection_path = f"artifacts/{app_id}/companies/{user_company}/team_polls"
                            success, response = add_document_to_firestore_collection(
                                collection_path,
                                {
                                    "question": poll_question,
                                    "options": options,
                                    "votes": {option: 0 for option in options}, # Initialize votes to 0
                                    "active": True,
                                    "created_by": current_username,
                                    "created_at": datetime.now()
                                },
                                FIREBASE_WEB_API_KEY,
                                FIRESTORE_BASE_URL
                            )
                            if success:
                                st.success("Poll created successfully!")
                                log_activity(f"created a new poll: '{poll_question}'.", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                                st.session_state.polls_needs_refresh = True
                                st.rerun()
                            else:
                                st.error(f"Error creating poll: {response}")
                                log_activity(f"Error creating poll: {response}", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                        except Exception as e:
                            st.error(f"An unexpected error occurred: {e}")
                else:
                    st.warning("Please provide a poll question and options.")

        st.markdown("---")
        st.subheader("Active Polls")

        if st.button("Refresh Polls", key="refresh_polls_button") or st.session_state.get('polls_needs_refresh', True):
            # Fetch all polls for the company and filter active ones in Python
            all_polls = fetch_collection_data(
                f"artifacts/{app_id}/companies/{user_company}/team_polls", # Company-specific path
                FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY,
                FIRESTORE_BASE_URL=FIRESTORE_BASE_URL,
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
                                doc_path = f"artifacts/{app_id}/companies/{user_company}/team_polls/{poll['id']}" # Company-specific path
                                
                                current_votes = poll.get('votes', {})
                                current_votes[selected_option] = current_votes.get(selected_option, 0) + 1
                                
                                # Only update the 'votes' field
                                payload = {"fields": {"votes": {"mapValue": {"fields": {
                                    k: {"integerValue": str(v)} for k, v in current_votes.items()
                                }}}}}
                                
                                success, response = save_document_to_firestore(
                                    doc_path,
                                    poll['id'],
                                    {"votes": current_votes}, # Send only the updated votes map
                                    FIREBASE_WEB_API_KEY,
                                    FIRESTORE_BASE_URL
                                )
                                if success:
                                    st.session_state[user_voted_key] = True # Mark user as voted in session
                                    st.success("Vote submitted successfully!")
                                    log_activity(f"voted on poll '{poll['question']}' for option '{selected_option}'.", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                                    st.session_state.polls_needs_refresh = True
                                    st.rerun() # Rerun to update the displayed results
                                else:
                                    st.error(f"Error submitting vote: {response}")
                                    log_activity(f"Error submitting vote: {response}", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                            except Exception as e:
                                st.error(f"An unexpected error occurred: {e}")
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
                            doc_path = f"artifacts/{app_id}/companies/{user_company}/team_polls/{poll['id']}" # Company-specific path
                            success, response = save_document_to_firestore(
                                doc_path,
                                poll['id'],
                                {"active": False}, # Only update active status
                                FIREBASE_WEB_API_KEY,
                                FIRESTORE_BASE_URL
                            )
                            if success:
                                st.success("Poll closed successfully!")
                                log_activity(f"closed poll: '{poll['question']}'.", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                                st.session_state.polls_needs_refresh = True
                                st.rerun()
                            else:
                                st.error(f"Error closing poll: {response}")
                                log_activity(f"Error closing poll: {response}", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                        except Exception as e:
                            st.error(f"An unexpected error occurred: {e}")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No active polls for this company. Be the first to create one!")

    with tab_activity:
        st.subheader("Recent Activity Feed")
        st.write("See a chronological log of recent actions within the Collaboration Hub across all users (public log).")

        if st.button("Refresh Activity Feed", key="refresh_activity_button") or st.session_state.get('activity_needs_refresh', True):
            st.session_state.firestore_activity_log = fetch_collection_data(
                f"artifacts/{app_id}/public/data/activity_feed", # This remains public
                FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY,
                FIRESTORE_BASE_URL=FIRESTORE_BASE_URL,
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
        st.subheader("Candidate Pipeline")
        st.info(f"The candidate pipeline for **{st.session_state.get('user_company', 'your company')}**.")
        st.write("Manage and track candidates through your company's hiring process.")

        # Load pipeline data for the current company
        if 'candidate_pipeline' not in st.session_state or st.session_state.get('pipeline_needs_refresh', True):
            st.session_state.candidate_pipeline = fetch_collection_data(
                f"artifacts/{app_id}/companies/{user_company}/candidate_pipeline",
                FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY,
                FIRESTORE_BASE_URL=FIRESTORE_BASE_URL,
                order_by_field="added_at", # Order by creation time
                limit=50
            )
            st.session_state.pipeline_needs_refresh = False

        st.markdown("---")
        st.subheader("Add New Candidate to Pipeline")
        with st.form("add_pipeline_candidate_form", clear_on_submit=True):
            cand_name = st.text_input("Candidate Name:", key="pipeline_cand_name")
            cand_email = st.text_input("Candidate Email:", key="pipeline_cand_email")
            cand_role = st.text_input("Applied Role:", key="pipeline_cand_role")
            cand_status = st.selectbox("Initial Status:", ["Application Received", "Screening", "Interview Scheduled", "Offer Extended", "Hired", "Rejected"], key="pipeline_cand_status")
            cand_notes = st.text_area("Initial Notes:", key="pipeline_cand_notes")
            add_pipeline_button = st.form_submit_button("Add Candidate")

            if add_pipeline_button:
                if cand_name and cand_email and cand_role:
                    try:
                        collection_path = f"artifacts/{app_id}/companies/{user_company}/candidate_pipeline"
                        success, response = add_document_to_firestore_collection(
                            collection_path,
                            {
                                "candidate_name": cand_name,
                                "candidate_email": cand_email,
                                "applied_role": cand_role,
                                "status": cand_status,
                                "notes": cand_notes,
                                "added_by": current_username,
                                "added_at": datetime.now(),
                                "last_updated": datetime.now()
                            },
                            FIREBASE_WEB_API_KEY,
                            FIRESTORE_BASE_URL
                        )
                        if success:
                            st.success(f"Candidate '{cand_name}' added to pipeline!")
                            log_activity(f"added candidate '{cand_name}' to pipeline.", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                            st.session_state.pipeline_needs_refresh = True
                            st.rerun()
                        else:
                            st.error(f"Error adding candidate: {response}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
                else:
                    st.warning("Please fill in candidate name, email, and applied role.")

        st.markdown("---")
        st.subheader("Current Candidate Pipeline")

        if st.button("Refresh Pipeline", key="refresh_pipeline_button"):
            st.session_state.pipeline_needs_refresh = True
            st.rerun()

        if st.session_state.candidate_pipeline:
            pipeline_df = pd.DataFrame(st.session_state.candidate_pipeline)
            # Sort by last_updated or added_at for display
            pipeline_df['display_last_updated'] = pipeline_df['last_updated'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M') if isinstance(x, datetime) else str(x))
            
            display_cols = ['candidate_name', 'applied_role', 'status', 'notes', 'added_by', 'display_last_updated']
            st.dataframe(pipeline_df[display_cols].rename(columns={'display_last_updated': 'Last Updated'}), use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("Update Candidate Status")
            # Create a dictionary for easy lookup of candidates by ID
            candidate_dict = {cand['id']: cand for cand in st.session_state.candidate_pipeline}
            
            candidate_to_update_id = st.selectbox(
                "Select Candidate to Update:",
                options=list(candidate_dict.keys()),
                format_func=lambda x: candidate_dict[x]['candidate_name'] + " (" + candidate_dict[x]['applied_role'] + ")",
                key="update_cand_select"
            )

            if candidate_to_update_id:
                selected_candidate = candidate_dict[candidate_to_update_id]
                st.write(f"Updating status for: **{selected_candidate['candidate_name']}** (Current Status: **{selected_candidate['status']}**)")
                
                new_status = st.selectbox(
                    "New Status:",
                    ["Application Received", "Screening", "Interview Scheduled", "Offer Extended", "Hired", "Rejected"],
                    index=["Application Received", "Screening", "Interview Scheduled", "Offer Extended", "Hired", "Rejected"].index(selected_candidate['status']),
                    key="new_status_select"
                )
                updated_notes = st.text_area("Update Notes (optional):", value=selected_candidate.get('notes', ''), key="update_notes_area")

                if st.button("Apply Status Update", key="apply_status_update_button"):
                    try:
                        doc_path = f"artifacts/{app_id}/companies/{user_company}/candidate_pipeline/{candidate_to_update_id}"
                        update_data = {
                            "status": new_status,
                            "notes": updated_notes,
                            "last_updated": datetime.now()
                        }
                        success, response = save_document_to_firestore(
                            doc_path,
                            candidate_to_update_id,
                            update_data,
                            FIREBASE_WEB_API_KEY,
                            FIRESTORE_BASE_URL
                        )
                        if success:
                            st.success(f"Status for '{selected_candidate['candidate_name']}' updated to '{new_status}'!")
                            log_activity(f"updated status for '{selected_candidate['candidate_name']}' to '{new_status}'.", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                            st.session_state.pipeline_needs_refresh = True
                            st.rerun()
                        else:
                            st.error(f"Error updating status: {response}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
            else:
                st.info("No candidates in pipeline to update.")
        else:
            st.info("No candidates in the pipeline yet. Add a new candidate above!")


    st.markdown("</div>", unsafe_allow_html=True)
