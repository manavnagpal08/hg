import streamlit as st
import requests
from datetime import datetime
import json
import pandas as pd

# --- Helper functions (copied from collaboration.py for self-containment) ---
def to_firestore_format(data: dict) -> dict:
    """Converts a Python dictionary to Firestore REST API 'fields' format."""
    fields = {}
    for key, value in data.items():
        if isinstance(value, str):
            fields[key] = {"stringValue": value}
        elif isinstance(value, int):
            fields[key] = {"integerValue": str(value)}
        elif isinstance(value, float):
            fields[key] = {"doubleValue": value}
        elif isinstance(value, bool):
            fields[key] = {"booleanValue": value}
        elif isinstance(value, datetime):
            fields[key] = {"timestampValue": value.isoformat() + "Z"}
        elif isinstance(value, list):
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
                elif isinstance(item, dict):
                    array_values.append({"mapValue": {"fields": to_firestore_format(item)['fields']}})
                else:
                    array_values.append({"stringValue": str(item)})
            fields[key] = {"arrayValue": {"values": array_values}}
        elif isinstance(value, dict):
            fields[key] = {"mapValue": {"fields": to_firestore_format(value)['fields']}}
        elif value is None:
            fields[key] = {"nullValue": None}
        else:
            fields[key] = {"stringValue": str(value)}
    return {"fields": fields}

def from_firestore_format(firestore_data: dict) -> dict:
    """Converts Firestore REST API 'fields' format to a Python dictionary."""
    data = {}
    if "fields" not in firestore_data:
        return data
    
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
            try:
                data[key] = datetime.fromisoformat(value_obj["timestampValue"].replace('Z', ''))
            except ValueError:
                data[key] = value_obj["timestampValue"]
        elif "arrayValue" in value_obj and "values" in value_obj["arrayValue"]:
            data[key] = [from_firestore_format({"fields": {"_": item}})["_"] if "mapValue" not in item else from_firestore_format({"fields": item["mapValue"]["fields"]}) for item in value_obj["arrayValue"]["values"]]
        elif "mapValue" in value_obj and "fields" in value_obj["mapValue"]:
            data[key] = from_firestore_format({"fields": value_obj["mapValue"]["fields"]})
        elif "nullValue" in value_obj:
            data[key] = None
    return data

def log_activity(message: str, user: str, FIREBASE_WEB_API_KEY: str, FIRESTORE_BASE_URL: str, app_id: str):
    """
    Logs an activity with a timestamp to Firestore (via REST API) and session state.
    This log is public/common across all companies.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {user}: {message}"

    if 'activity_log' not in st.session_state:
        st.session_state.activity_log = []
    st.session_state.activity_log.insert(0, log_entry)
    st.session_state.activity_log = st.session_state.activity_log[:50]

    try:
        collection_url = f"{FIRESTORE_BASE_URL}/artifacts/{app_id}/public/data/activity_feed"
        payload = to_firestore_format({
            "message": message,
            "user": user,
            "timestamp": datetime.now()
        })
        response = requests.post(collection_url, json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error logging activity to Firestore via REST API: {e}")

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
                doc_id = doc_entry['name'].split('/')[-1]
                doc_data = from_firestore_format(doc_entry)
                doc_data['id'] = doc_id
                documents.append(doc_data)
        
        if order_by_field and all(order_by_field in doc for doc in documents):
            documents.sort(key=lambda x: x.get(order_by_field), reverse=True)
        
        return documents[:limit]

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from {collection_path} via REST API: {e}")
        return []

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

def update_document_in_firestore(collection_path, doc_id, data, api_key, base_url):
    """Updates a document in Firestore using PATCH."""
    url = f"{base_url}/{collection_path}/{doc_id}?key={api_key}"
    update_mask_fields = ",".join(data.keys())
    url_with_mask = f"{url}&updateMask.fieldPaths={update_mask_fields}"
    
    firestore_data = to_firestore_format(data)
    try:
        res = requests.patch(url_with_mask, json=firestore_data)
        res.raise_for_status()
        return True, res.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Firestore update error: {e}")
        return False, str(e)

def delete_document_from_firestore(collection_path, doc_id, api_key, base_url):
    """Deletes a document from Firestore."""
    url = f"{base_url}/{collection_path}/{doc_id}?key={api_key}"
    try:
        res = requests.delete(url)
        res.raise_for_status()
        return True, res.text
    except requests.exceptions.RequestException as e:
        st.error(f"Firestore delete error: {e}")
        return False, str(e)

# --- Main Onboarding/Offboarding Page Function ---
def onboarding_offboarding_page(app_id: str, FIREBASE_WEB_API_KEY: str, FIRESTORE_BASE_URL: str):
    st.markdown('<div class="dashboard-header">🚀 Onboarding & Offboarding</div>', unsafe_allow_html=True)
    st.write("Streamline the process of welcoming new hires and managing employee departures with customizable checklists.")

    current_username = st.session_state.get('username', 'Anonymous User')
    user_company = st.session_state.get('user_company', 'default_company').replace(' ', '_').lower()

    st.info(f"You are managing processes for: **{st.session_state.get('user_company', 'N/A')}**")
    st.write(f"**DEBUG: Current Company for Data Isolation:** `{user_company}`")
    st.markdown("---")

    # Initialize session state for processes
    if 'onboarding_offboarding_processes' not in st.session_state:
        st.session_state.onboarding_offboarding_processes = []
    if 'processes_needs_refresh' not in st.session_state:
        st.session_state.processes_needs_refresh = True

    # Tabs for creating and viewing processes
    tab_create, tab_view = st.tabs(["➕ Create New Process", "📋 View & Manage Processes"])

    with tab_create:
        st.subheader("Create New Onboarding or Offboarding Process")
        process_type = st.radio("Select Process Type:", ["Onboarding", "Offboarding"], key="process_type_radio")
        
        with st.form("create_process_form", clear_on_submit=True):
            target_employee_name = st.text_input("Employee Name:", help="Name of the employee this process is for.", key="target_employee_name_input")
            process_title = st.text_input("Process Title:", value=f"{process_type} Checklist for {target_employee_name}" if target_employee_name else "", key="process_title_input")
            due_date = st.date_input("Target Completion Date:", key="process_due_date", value=datetime.now().date())
            
            st.markdown("---")
            st.markdown("##### Define Tasks for this Process")
            st.info("Add tasks one by one. You can assign them to roles/departments.")

            # Dynamic task input
            if 'temp_tasks' not in st.session_state:
                st.session_state.temp_tasks = []

            task_description = st.text_input("Task Description:", key="task_desc_input")
            task_assigned_to = st.text_input("Assigned To (e.g., HR, IT, Manager, Employee):", key="task_assigned_to_input")
            
            if st.button("Add Task to List", key="add_task_to_list_button"):
                if task_description and task_assigned_to:
                    st.session_state.temp_tasks.append({
                        "description": task_description,
                        "assigned_to": task_assigned_to,
                        "status": "Pending",
                        "due_date": str(due_date) # Use the process due date as default for tasks
                    })
                    st.success("Task added to list. Add more or create process.")
                else:
                    st.warning("Please provide task description and assignee.")
            
            st.markdown("---")
            st.markdown("##### Current Tasks in List:")
            if st.session_state.temp_tasks:
                for i, task in enumerate(st.session_state.temp_tasks):
                    st.write(f"- **{task['description']}** (Assigned to: {task['assigned_to']}, Status: {task['status']})")
                if st.button("Clear Task List", key="clear_task_list_button"):
                    st.session_state.temp_tasks = []
                    st.rerun()
            else:
                st.info("No tasks added yet. Add tasks above.")

            st.markdown("---")
            create_process_button = st.form_submit_button("Create Process")

            if create_process_button:
                if target_employee_name and process_title and st.session_state.temp_tasks:
                    try:
                        collection_path = f"artifacts/{app_id}/companies/{user_company}/onboarding_offboarding"
                        process_data = {
                            "employee_name": target_employee_name,
                            "process_type": process_type,
                            "title": process_title,
                            "due_date": str(due_date),
                            "tasks": st.session_state.temp_tasks, # Store the list of tasks
                            "created_by": current_username,
                            "created_at": datetime.now(),
                            "overall_status": "In Progress" # Initial status
                        }
                        success, response_data = add_document_to_firestore_collection(
                            collection_path, process_data, FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL
                        )
                        if success:
                            st.success(f"{process_type} process for '{target_employee_name}' created successfully!")
                            log_activity(f"created a new {process_type} process for '{target_employee_name}' in {st.session_state.get('user_company', 'their company')}.", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                            st.session_state.temp_tasks = [] # Clear tasks after submission
                            st.session_state.processes_needs_refresh = True
                            st.rerun()
                        else:
                            st.error(f"Error creating process: {response_data}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
                else:
                    st.warning("Please fill in employee name, process title, and add at least one task.")

    with tab_view:
        st.subheader("Current Onboarding & Offboarding Processes")
        
        filter_status = st.selectbox("Filter by Overall Status:", ["All", "In Progress", "Completed", "Overdue"], key="process_filter_status")
        
        if st.button("Refresh Processes", key="refresh_processes_button") or st.session_state.processes_needs_refresh:
            st.session_state.onboarding_offboarding_processes = fetch_collection_data(
                f"artifacts/{app_id}/companies/{user_company}/onboarding_offboarding",
                FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY,
                FIRESTORE_BASE_URL=FIRESTORE_BASE_URL,
                order_by_field="created_at",
                limit=50
            )
            st.session_state.processes_needs_refresh = False

        filtered_processes = []
        for process in st.session_state.onboarding_offboarding_processes:
            process_overall_status = process.get("overall_status", "In Progress")
            
            # Check for overdue
            if process_overall_status == "In Progress" and 'due_date' in process:
                try:
                    process_due_date = datetime.strptime(process['due_date'], '%Y-%m-%d').date()
                    if process_due_date < datetime.now().date():
                        process_overall_status = "Overdue"
                except ValueError:
                    pass # Malformed date, keep original status

            if filter_status == "All" or process_overall_status == filter_status:
                filtered_processes.append(process)

        if filtered_processes:
            for process in filtered_processes:
                overall_status = process.get("overall_status", "In Progress")
                if overall_status == "In Progress" and 'due_date' in process:
                    try:
                        process_due_date = datetime.strptime(process['due_date'], '%Y-%m-%d').date()
                        if process_due_date < datetime.now().date():
                            overall_status = "Overdue"
                    except ValueError:
                        pass # Malformed date

                status_color = "orange"
                if overall_status == "Completed":
                    status_color = "green"
                elif overall_status == "Overdue":
                    status_color = "red"

                st.markdown(f"""
                <div style="background-color: {'#3A3A3A' if st.session_state.get('dark_mode_main') else '#f0f2f6'}; padding: 15px; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <h4 style="color: {'#00cec9' if st.session_state.get('dark_mode_main') else '#00cec9'}; margin-bottom: 5px;">
                        {process.get('title', 'No Title')} ({process.get('process_type', 'N/A')})
                    </h4>
                    <p style="font-size: 0.9em; color: {'#BBBBBB' if st.session_state.get('dark_mode_main') else '#666'}; margin-bottom: 5px;">
                        Employee: **{process.get('employee_name', 'N/A')}** | Due: **{process.get('due_date', 'N/A')}** | Status: <span style="color: {status_color}; font-weight: bold;">{overall_status.upper()}</span>
                    </p>
                """, unsafe_allow_html=True)

                st.markdown("##### Tasks:")
                tasks_df = pd.DataFrame(process.get('tasks', []))
                if not tasks_df.empty:
                    st.dataframe(tasks_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No tasks defined for this process.")

                if overall_status != "Completed":
                    # Allow updating individual task status
                    st.markdown("---")
                    st.write("Update Task Status:")
                    task_options = {i: task['description'] for i, task in enumerate(process.get('tasks', []))}
                    if task_options:
                        selected_task_idx = st.selectbox(
                            "Select a task:",
                            options=list(task_options.keys()),
                            format_func=lambda x: task_options[x],
                            key=f"task_select_{process['id']}"
                        )
                        if selected_task_idx is not None:
                            current_task = process['tasks'][selected_task_idx]
                            new_task_status = st.selectbox(
                                "New Status:",
                                ["Pending", "In Progress", "Completed", "Blocked"],
                                index=["Pending", "In Progress", "Completed", "Blocked"].index(current_task.get('status', 'Pending')),
                                key=f"new_task_status_{process['id']}_{selected_task_idx}"
                            )
                            if st.button(f"Update Task '{current_task['description']}'", key=f"update_task_button_{process['id']}_{selected_task_idx}"):
                                process['tasks'][selected_task_idx]['status'] = new_task_status
                                
                                # Check if all tasks are completed
                                all_tasks_completed = all(t['status'] == 'Completed' for t in process['tasks'])
                                new_overall_status = "Completed" if all_tasks_completed else "In Progress"

                                try:
                                    doc_id = process['id']
                                    collection_path = f"artifacts/{app_id}/companies/{user_company}/onboarding_offboarding"
                                    update_data = {
                                        "tasks": process['tasks'], # Update the entire tasks list
                                        "overall_status": new_overall_status,
                                        "last_updated": datetime.now()
                                    }
                                    success, response_data = update_document_in_firestore(
                                        collection_path, doc_id, update_data, FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL
                                    )
                                    if success:
                                        st.success("Task status updated!")
                                        log_activity(f"updated task '{current_task['description']}' in {process.get('title', 'a process')} to '{new_task_status}'.", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                                        st.session_state.processes_needs_refresh = True
                                        st.rerun()
                                    else:
                                        st.error(f"Error updating task: {response_data}")
                                except Exception as e:
                                    st.error(f"An unexpected error occurred: {e}")
                    else:
                        st.info("No tasks to update for this process.")
                
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No processes found for this company. Create a new process using the 'Create New Process' tab.")

