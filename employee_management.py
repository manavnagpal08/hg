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
    # Construct updateMask for partial updates
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

# --- Main Employee Management Page Function ---
def employee_management_page(app_id: str, FIREBASE_WEB_API_KEY: str, FIRESTORE_BASE_URL: str):
    st.markdown('<div class="dashboard-header">👥 Employee Management (HRIS)</div>', unsafe_allow_html=True)
    st.write("Manage comprehensive profiles for all your company's employees.")

    current_username = st.session_state.get('username', 'Anonymous User')
    user_company = st.session_state.get('user_company', 'default_company').replace(' ', '_').lower()

    st.info(f"You are managing employees for: **{st.session_state.get('user_company', 'N/A')}**")
    st.write(f"**DEBUG: Current Company for Data Isolation:** `{user_company}`")
    st.markdown("---")

    # Initialize session state for employees if not present
    if 'employees' not in st.session_state:
        st.session_state.employees = []
    if 'employees_needs_refresh' not in st.session_state:
        st.session_state.employees_needs_refresh = True

    # Tabs for Add/View Employees
    tab_add, tab_view = st.tabs(["➕ Add New Employee", "📋 View All Employees"])

    with tab_add:
        st.subheader("Add New Employee Profile")
        with st.form("add_employee_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                emp_name = st.text_input("Full Name:", key="emp_name_input")
                emp_email = st.text_input("Email (Unique Identifier):", key="emp_email_input")
                emp_phone = st.text_input("Phone Number:", key="emp_phone_input")
                emp_address = st.text_area("Address:", key="emp_address_input")
                emp_dob = st.date_input("Date of Birth:", key="emp_dob_input", value=datetime(2000, 1, 1).date())
            with col2:
                emp_job_title = st.text_input("Job Title:", key="emp_job_title_input")
                emp_department = st.text_input("Department:", key="emp_department_input")
                emp_manager = st.text_input("Manager (Name/Email):", key="emp_manager_input")
                emp_start_date = st.date_input("Start Date:", key="emp_start_date_input", value=datetime.now().date())
                emp_status = st.selectbox("Employment Status:", ["Active", "On Leave", "Terminated"], key="emp_status_input")
            
            add_employee_button = st.form_submit_button("Add Employee")

            if add_employee_button:
                if emp_name and emp_email and emp_job_title and emp_department:
                    if "@" not in emp_email or "." not in emp_email:
                        st.error("Please enter a valid email address.")
                    else:
                        try:
                            # Use email as document ID (sanitized)
                            doc_id = emp_email.replace('.', '_').replace('@', '_')
                            collection_path = f"artifacts/{app_id}/companies/{user_company}/employees"
                            
                            # Check if employee already exists
                            check_url = f"{FIRESTORE_BASE_URL}/{collection_path}/{doc_id}?key={FIREBASE_WEB_API_KEY}"
                            check_response = requests.get(check_url)
                            if check_response.status_code == 200:
                                st.warning(f"An employee with email '{emp_email}' already exists in this company.")
                            else:
                                employee_data = {
                                    "name": emp_name,
                                    "email": emp_email,
                                    "phone": emp_phone,
                                    "address": emp_address,
                                    "dob": str(emp_dob),
                                    "job_title": emp_job_title,
                                    "department": emp_department,
                                    "manager": emp_manager,
                                    "start_date": str(emp_start_date),
                                    "status": emp_status,
                                    "added_by": current_username,
                                    "added_at": datetime.now(),
                                    "last_updated": datetime.now()
                                }
                                success, response_data = update_document_in_firestore(
                                    collection_path, doc_id, employee_data, FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL
                                ) # Using update_document to create/patch by ID
                                if success:
                                    st.success(f"Employee '{emp_name}' added successfully!")
                                    log_activity(f"added new employee '{emp_name}' to {st.session_state.get('user_company', 'their company')}'s HRIS.", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                                    st.session_state.employees_needs_refresh = True
                                    st.rerun()
                                else:
                                    st.error(f"Error adding employee: {response_data}")
                        except Exception as e:
                            st.error(f"An unexpected error occurred: {e}")
                else:
                    st.warning("Please fill in all required fields (Name, Email, Job Title, Department).")

    with tab_view:
        st.subheader("All Employees")
        if st.button("Refresh Employee List", key="refresh_employees_button") or st.session_state.employees_needs_refresh:
            st.session_state.employees = fetch_collection_data(
                f"artifacts/{app_id}/companies/{user_company}/employees",
                FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY,
                FIRESTORE_BASE_URL=FIRESTORE_BASE_URL,
                order_by_field="name",
                limit=100
            )
            st.session_state.employees_needs_refresh = False

        if st.session_state.employees:
            employees_df = pd.DataFrame(st.session_state.employees)
            
            # Convert datetime objects to string for display if needed
            for col in ['added_at', 'last_updated']:
                if col in employees_df.columns:
                    employees_df[col] = employees_df[col].apply(lambda x: x.strftime("%Y-%m-%d %H:%M") if isinstance(x, datetime) else str(x))

            display_cols = [
                'name', 'email', 'job_title', 'department', 'manager', 'status', 
                'start_date', 'phone', 'address', 'dob', 'added_by', 'last_updated'
            ]
            # Filter to only display columns that actually exist in the DataFrame
            display_cols = [col for col in display_cols if col in employees_df.columns]

            st.dataframe(employees_df[display_cols], use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("Update / Delete Employee")

            employee_options = {emp['id']: emp['name'] for emp in st.session_state.employees}
            selected_employee_id = st.selectbox(
                "Select Employee to Update or Delete:",
                options=list(employee_options.keys()),
                format_func=lambda x: employee_options[x],
                key="select_employee_for_action"
            )

            if selected_employee_id:
                selected_employee = next((emp for emp in st.session_state.employees if emp['id'] == selected_employee_id), None)
                if selected_employee:
                    st.write(f"Editing: **{selected_employee['name']}**")
                    with st.form(f"edit_employee_form_{selected_employee_id}", clear_on_submit=False):
                        col1_edit, col2_edit = st.columns(2)
                        with col1_edit:
                            edited_name = st.text_input("Full Name:", value=selected_employee.get('name', ''), key=f"edit_emp_name_{selected_employee_id}")
                            edited_email = st.text_input("Email:", value=selected_employee.get('email', ''), disabled=True, help="Email cannot be changed as it's the unique ID.", key=f"edit_emp_email_{selected_employee_id}")
                            edited_phone = st.text_input("Phone Number:", value=selected_employee.get('phone', ''), key=f"edit_emp_phone_{selected_employee_id}")
                            edited_address = st.text_area("Address:", value=selected_employee.get('address', ''), key=f"edit_emp_address_{selected_employee_id}")
                            # Ensure date is a date object for date_input
                            edited_dob_val = datetime.strptime(selected_employee['dob'], '%Y-%m-%d').date() if isinstance(selected_employee.get('dob'), str) else selected_employee.get('dob', datetime(2000, 1, 1).date())
                            edited_dob = st.date_input("Date of Birth:", value=edited_dob_val, key=f"edit_emp_dob_{selected_employee_id}")
                        with col2_edit:
                            edited_job_title = st.text_input("Job Title:", value=selected_employee.get('job_title', ''), key=f"edit_emp_job_title_{selected_employee_id}")
                            edited_department = st.text_input("Department:", value=selected_employee.get('department', ''), key=f"edit_emp_department_{selected_employee_id}")
                            edited_manager = st.text_input("Manager:", value=selected_employee.get('manager', ''), key=f"edit_emp_manager_{selected_employee_id}")
                            # Ensure date is a date object for date_input
                            edited_start_date_val = datetime.strptime(selected_employee['start_date'], '%Y-%m-%d').date() if isinstance(selected_employee.get('start_date'), str) else selected_employee.get('start_date', datetime.now().date())
                            edited_start_date = st.date_input("Start Date:", value=edited_start_date_val, key=f"edit_emp_start_date_{selected_employee_id}")
                            edited_status = st.selectbox("Employment Status:", ["Active", "On Leave", "Terminated"], index=["Active", "On Leave", "Terminated"].index(selected_employee.get('status', 'Active')), key=f"edit_emp_status_{selected_employee_id}")
                        
                        col_actions = st.columns(2)
                        with col_actions[0]:
                            update_button = st.form_submit_button("Update Employee Details")
                        with col_actions[1]:
                            delete_button = st.form_submit_button("Delete Employee", help="This action cannot be undone.")

                        if update_button:
                            try:
                                updated_data = {
                                    "name": edited_name,
                                    "phone": edited_phone,
                                    "address": edited_address,
                                    "dob": str(edited_dob),
                                    "job_title": edited_job_title,
                                    "department": edited_department,
                                    "manager": edited_manager,
                                    "start_date": str(edited_start_date),
                                    "status": edited_status,
                                    "last_updated": datetime.now()
                                }
                                success, response_data = update_document_in_firestore(
                                    collection_path, selected_employee_id, updated_data, FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL
                                )
                                if success:
                                    st.success(f"Employee '{edited_name}' updated successfully!")
                                    log_activity(f"updated employee '{edited_name}' in {st.session_state.get('user_company', 'their company')}'s HRIS.", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                                    st.session_state.employees_needs_refresh = True
                                    st.rerun()
                                else:
                                    st.error(f"Error updating employee: {response_data}")
                            except Exception as e:
                                st.error(f"An unexpected error occurred during update: {e}")

                        if delete_button:
                            # Implement a confirmation step if this were a real app
                            if st.warning(f"Are you sure you want to delete {selected_employee['name']}? This cannot be undone."):
                                if st.button(f"Confirm Delete {selected_employee['name']}", key=f"confirm_delete_{selected_employee_id}"):
                                    try:
                                        success, response_data = delete_document_from_firestore(
                                            collection_path, selected_employee_id, FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL
                                        )
                                        if success:
                                            st.success(f"Employee '{selected_employee['name']}' deleted successfully!")
                                            log_activity(f"deleted employee '{selected_employee['name']}' from {st.session_state.get('user_company', 'their company')}'s HRIS.", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                                            st.session_state.employees_needs_refresh = True
                                            st.rerun()
                                        else:
                                            st.error(f"Error deleting employee: {response_data}")
                                    except Exception as e:
                                        st.error(f"An unexpected error occurred during delete: {e}")
                else:
                    st.info("Select an employee from the dropdown to view/edit their details.")
            else:
                st.info("No employees to select. Add new employees using the 'Add New Employee' tab.")
        else:
            st.info("No employees added yet for this company. Add new employees using the 'Add New Employee' tab.")

