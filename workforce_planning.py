import streamlit as st
import requests
from datetime import datetime
import json
import pandas as pd

# --- Helper functions (copied for self-containment) ---
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

# --- Main Workforce Planning Page Function ---
def workforce_planning_page(app_id: str, FIREBASE_WEB_API_KEY: str, FIRESTORE_BASE_URL: str):
    st.markdown('<div class="dashboard-header">📈 Workforce Planning & Succession Management</div>', unsafe_allow_html=True)
    st.write("Strategically plan for future talent needs and identify potential successors for key roles within your company.")

    current_username = st.session_state.get('username', 'Anonymous User')
    user_company = st.session_state.get('user_company', 'default_company').replace(' ', '_').lower()

    st.info(f"You are managing workforce planning for: **{st.session_state.get('user_company', 'N/A')}**")
    st.write(f"**DEBUG: Current Company for Data Isolation:** `{user_company}`")
    st.markdown("---")

    # Tabs for different features
    tab_demand, tab_supply, tab_succession, tab_skill_gap = st.tabs([
        "📊 Demand Forecasting (Mock)", "👥 Supply Analysis (Mock)", "👑 Succession Planning", "🧠 Skill Gap Analysis (Mock)"
    ])

    with tab_demand:
        st.subheader("Demand Forecasting (Mock)")
        st.info(f"Predict future talent needs based on business growth and turnover. This is a mock feature; a full implementation would involve advanced analytics and integration with business data. Data is isolated for **{st.session_state.get('user_company', 'your company')}**.")

        # Initialize session state for demand forecasts
        if 'demand_forecasts' not in st.session_state:
            st.session_state.demand_forecasts = []
        if 'demand_forecasts_needs_refresh' not in st.session_state:
            st.session_state.demand_forecasts_needs_refresh = True

        with st.form("demand_forecast_form", clear_on_submit=True):
            role_needed = st.text_input("Role Needed:", key="df_role")
            num_positions = st.number_input("Number of Positions:", min_value=1, value=1, key="df_num_pos")
            target_quarter = st.selectbox("Target Quarter:", ["Q1", "Q2", "Q3", "Q4"], key="df_quarter")
            target_year = st.number_input("Target Year:", min_value=datetime.now().year, value=datetime.now().year, key="df_year")
            add_forecast_button = st.form_submit_button("Add Forecast")

            if add_forecast_button:
                if role_needed and num_positions:
                    try:
                        collection_path = f"artifacts/{app_id}/companies/{user_company}/workforce_planning/demand_forecasts"
                        forecast_data = {
                            "role": role_needed,
                            "positions": num_positions,
                            "quarter": target_quarter,
                            "year": target_year,
                            "created_by": current_username,
                            "created_at": datetime.now()
                        }
                        success, response_data = add_document_to_firestore_collection(
                            collection_path, forecast_data, FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL
                        )
                        if success:
                            st.success(f"Demand forecast for {num_positions} {role_needed}(s) in {target_quarter} {target_year} added!")
                            log_activity(f"added demand forecast for {role_needed} in {target_quarter} {target_year}.", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                            st.session_state.demand_forecasts_needs_refresh = True
                            st.rerun()
                        else:
                            st.error(f"Error adding forecast: {response_data}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
                else:
                    st.warning("Please provide role and number of positions.")
        
        st.markdown("---")
        st.subheader("Current Demand Forecasts")

        if st.button("Refresh Demand Forecasts", key="refresh_df_button") or st.session_state.demand_forecasts_needs_refresh:
            st.session_state.demand_forecasts = fetch_collection_data(
                f"artifacts/{app_id}/companies/{user_company}/workforce_planning/demand_forecasts",
                FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY,
                FIRESTORE_BASE_URL=FIRESTORE_BASE_URL,
                order_by_field="created_at",
                limit=20
            )
            st.session_state.demand_forecasts_needs_refresh = False

        if st.session_state.demand_forecasts:
            forecast_df = pd.DataFrame(st.session_state.demand_forecasts)
            forecast_df['Target Period'] = forecast_df['quarter'] + ' ' + forecast_df['year'].astype(str)
            display_cols = ['role', 'positions', 'Target Period', 'created_by', 'created_at']
            st.dataframe(forecast_df[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No demand forecasts added yet. Add some above!")

    with tab_supply:
        st.subheader("Supply Analysis (Mock)")
        st.info(f"Analyze your current workforce to understand available talent. This is a mock feature; a full implementation would integrate with your HRIS. Data is isolated for **{st.session_state.get('user_company', 'your company')}**.")
        
        st.write("Mock data representing your current employee demographics and skills:")

        mock_supply_data = {
            "Department": ["Engineering", "Sales", "Marketing", "HR", "Engineering", "Sales", "Engineering"],
            "Employees": [50, 30, 20, 10, 45, 25, 55],
            "Avg. Tenure (Years)": [4.5, 3.2, 2.8, 5.1, 4.0, 3.5, 4.8],
            "Key Skills": [
                "Python, Java, Cloud", "CRM, Negotiation", "SEO, Content", "Recruitment, Policy",
                "Node.js, React", "Lead Gen, Closing", "DevOps, AWS"
            ]
        }
        supply_df = pd.DataFrame(mock_supply_data)
        st.dataframe(supply_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Employee Distribution by Department (Mock Chart)")
        dept_counts = supply_df.groupby('Department')['Employees'].sum().reset_index()
        fig_dept_dist = px.pie(
            dept_counts,
            values='Employees',
            names='Department',
            title='Employee Distribution by Department',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_dept_dist, use_container_width=True)

    with tab_succession:
        st.subheader("Succession Planning")
        st.info(f"Identify and develop potential successors for critical roles. Data is isolated for **{st.session_state.get('user_company', 'your company')}**.")

        # Initialize session state for succession plans
        if 'succession_plans' not in st.session_state:
            st.session_state.succession_plans = []
        if 'succession_plans_needs_refresh' not in st.session_state:
            st.session_state.succession_plans_needs_refresh = True

        with st.form("succession_plan_form", clear_on_submit=True):
            critical_role = st.text_input("Critical Role:", help="e.g., 'Head of Engineering', 'VP Sales'", key="sp_role")
            incumbent_name = st.text_input("Current Incumbent:", key="sp_incumbent")
            
            st.markdown("##### Potential Successors (Add up to 3)")
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                successor1_name = st.text_input("Successor 1 Name:", key="s1_name")
                successor1_readiness = st.selectbox("Readiness:", ["High", "Medium", "Low"], key="s1_readiness")
            with col_s2:
                successor2_name = st.text_input("Successor 2 Name (Optional):", key="s2_name")
                successor2_readiness = st.selectbox("Readiness:", ["High", "Medium", "Low", "N/A"], index=3, key="s2_readiness")
            with col_s3:
                successor3_name = st.text_input("Successor 3 Name (Optional):", key="s3_name")
                successor3_readiness = st.selectbox("Readiness:", ["High", "Medium", "Low", "N/A"], index=3, key="s3_readiness")
            
            development_notes = st.text_area("Development Notes for Successors:", key="sp_dev_notes")
            add_plan_button = st.form_submit_button("Add Succession Plan")

            if add_plan_button:
                if critical_role and incumbent_name and successor1_name:
                    successors = []
                    if successor1_name:
                        successors.append({"name": successor1_name, "readiness": successor1_readiness})
                    if successor2_name:
                        successors.append({"name": successor2_name, "readiness": successor2_readiness})
                    if successor3_name:
                        successors.append({"name": successor3_name, "readiness": successor3_readiness})

                    try:
                        collection_path = f"artifacts/{app_id}/companies/{user_company}/workforce_planning/succession_plans"
                        plan_data = {
                            "critical_role": critical_role,
                            "incumbent": incumbent_name,
                            "successors": successors,
                            "development_notes": development_notes,
                            "created_by": current_username,
                            "created_at": datetime.now()
                        }
                        success, response_data = add_document_to_firestore_collection(
                            collection_path, plan_data, FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL
                        )
                        if success:
                            st.success(f"Succession plan for '{critical_role}' added successfully!")
                            log_activity(f"added succession plan for '{critical_role}'.", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                            st.session_state.succession_plans_needs_refresh = True
                            st.rerun()
                        else:
                            st.error(f"Error adding plan: {response_data}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
                else:
                    st.warning("Please provide critical role, incumbent, and at least one successor.")
        
        st.markdown("---")
        st.subheader("Current Succession Plans")

        if st.button("Refresh Succession Plans", key="refresh_sp_button") or st.session_state.succession_plans_needs_refresh:
            st.session_state.succession_plans = fetch_collection_data(
                f"artifacts/{app_id}/companies/{user_company}/workforce_planning/succession_plans",
                FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY,
                FIRESTORE_BASE_URL=FIRESTORE_BASE_URL,
                order_by_field="created_at",
                limit=20
            )
            st.session_state.succession_plans_needs_refresh = False

        if st.session_state.succession_plans:
            for plan in st.session_state.succession_plans:
                timestamp_obj = plan.get('created_at')
                timestamp_str = timestamp_obj.strftime("%Y-%m-%d %H:%M") if isinstance(timestamp_obj, datetime) else str(timestamp_obj)
                
                successors_display = ", ".join([f"{s['name']} ({s['readiness']})" for s in plan.get('successors', [])])

                st.markdown(f"""
                <div style="background-color: {'#3A3A3A' if st.session_state.get('dark_mode_main') else '#f0f2f6'}; padding: 15px; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <h5 style="color: {'#00cec9' if st.session_state.get('dark_mode_main') else '#00cec9'}; margin-bottom: 5px;">
                        Critical Role: **{plan.get('critical_role', 'N/A')}**
                    </h5>
                    <p style="font-size: 0.9em; color: {'#BBBBBB' if st.session_state.get('dark_mode_main') else '#666'}; margin-bottom: 5px;">
                        Current Incumbent: **{plan.get('incumbent', 'N/A')}**
                    </p>
                    <p style="font-size: 1.0em; color: {'#E0E0E0' if st.session_state.get('dark_mode_main') else '#333'};">
                        Potential Successors: {successors_display if successors_display else "None identified"}
                    </p>
                    <p style="font-size: 1.0em; color: {'#E0E0E0' if st.session_state.get('dark_mode_main') else '#333'};">
                        Development Notes: {plan.get('development_notes', 'No notes.')}
                    </p>
                    <p style="font-size: 0.8em; color: {'#999999' if st.session_state.get('dark_mode_main') else '#888'}; margin-top: 5px;">
                        Added by: {plan.get('created_by', 'Unknown')} at {timestamp_str}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No succession plans added yet. Create one above!")

    with tab_skill_gap:
        st.subheader("Skill Gap Analysis (Mock)")
        st.info(f"Identify discrepancies between current employee skills and future business needs. This is a mock feature; a full implementation would require comprehensive skill inventories and predictive modeling. Data is isolated for **{st.session_state.get('user_company', 'your company')}**.")
        
        st.write("Mock analysis of skill gaps in your company:")

        mock_skill_gap_data = {
            "Skill Area": ["AI/ML Engineering", "Cloud Security", "Data Science (Advanced)", "UX/UI Design", "Cybersecurity", "Sales Leadership"],
            "Current Proficiency (Avg)": [3.5, 2.8, 3.0, 4.2, 2.5, 3.8],
            "Required Proficiency (Avg)": [4.5, 4.0, 4.5, 4.0, 4.0, 4.5],
            "Gap (Required - Current)": [1.0, 1.2, 1.5, -0.2, 1.5, 0.7],
            "Priority": ["High", "High", "High", "Low", "High", "Medium"]
        }
        skill_gap_df = pd.DataFrame(mock_skill_gap_data)
        st.dataframe(skill_gap_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Skill Gap Visualization (Mock Chart)")
        
        fig_skill_gap = px.bar(
            skill_gap_df,
            x='Gap (Required - Current)',
            y='Skill Area',
            orientation='h',
            color='Priority',
            color_discrete_map={"High": "red", "Medium": "orange", "Low": "green"},
            title='Identified Skill Gaps',
            labels={'Gap (Required - Current)': 'Skill Gap (Required - Current Proficiency)', 'Skill Area': 'Skill Area'}
        )
        st.plotly_chart(fig_skill_gap, use_container_width=True)

