import streamlit as st
import pandas as pd
import plotly.express as px
import collections
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import smtplib # For sending emails via SMTP
from email.mime.text import MIMEText # For creating email messages

# --- Logging Function (can be shared or imported from main.py if needed) ---
def log_user_action(user_email, action, details=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if details:
        print(f"LOG: [{timestamp}] User '{user_email}' performed action '{action}' with details: {details}")
    else:
        print(f"LOG: [{timestamp}] User '{user_email}' performed action '{action}'")

# --- Firebase REST API Helper Functions (adapted from main.py) ---

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

def load_collection_from_firestore(collection_path, api_key, base_url):
    """Loads all documents from a Firestore collection."""
    url = f"{base_url}/{collection_path}?key={api_key}"
    try:
        res = requests.get(url)
        res.raise_for_status()
        docs_data = []
        if 'documents' in res.json():
            for doc in res.json()['documents']:
                doc_id = doc['name'].split('/')[-1]
                data = from_firestore_format(doc)
                data['id'] = doc_id # Add document ID to the data
                docs_data.append(data)
        return True, docs_data
    except requests.exceptions.RequestException as e:
        if e.response and e.response.status_code == 404:
            return True, [] # Collection not found, return empty list
        st.error(f"Firestore load error: {e}")
        return False, str(e)

# --- Email Sending Function using Gmail App Password ---
def send_actual_email(to_email, subject, body, gmail_address, gmail_app_password):
    """
    Attempts to send a real email via Gmail's SMTP server using an App Password.
    """
    if not gmail_address or not gmail_app_password:
        st.warning("Email sending skipped: Gmail Address or App Password not configured internally.")
        return False, "Gmail credentials not configured."

    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = gmail_address
        msg['To'] = to_email

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp: # Use SSL for port 465
        # For port 587 (TLS), use:
        # with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
        #     smtp.starttls()
            smtp.login(gmail_address, gmail_app_password)
            smtp.send_message(msg)
        
        st.success(f"📧 Real email sent to {to_email} via Gmail!")
        return True, "Email sent successfully."
    except smtplib.SMTPAuthenticationError:
        st.error("❌ Gmail Authentication Error: Please check your hardcoded Gmail address and App Password. Ensure 2-Step Verification is enabled and you've generated an App Password.")
        return False, "Authentication failed."
    except smtplib.SMTPServerDisconnected:
        st.error("❌ Gmail SMTP Server Disconnected: This might be a temporary network issue or a security block. Try again later.")
        return False, "Server disconnected."
    except Exception as e:
        st.error(f"❌ Failed to send real email to {to_email} due to an unexpected error: {e}")
        return False, str(e)


# --- Mock Salary Data (More Realistic and Granular) ---
MOCK_SALARY_DATA = [
    # Software Engineer - Bengaluru (Annual Salaries in INR Lakhs)
    {"role": "Software Engineer", "seniority": "Junior", "location": "Bengaluru, India", "min_exp": 0, "max_exp": 1, "min_salary": 400000, "max_salary": 600000, "avg_bonus_pct": 5, "avg_equity_pct": 0},
    {"role": "Software Engineer", "seniority": "Mid", "location": "Bengaluru, India", "min_exp": 2, "max_exp": 4, "min_salary": 800000, "max_salary": 1300000, "avg_bonus_pct": 8, "avg_equity_pct": 5},
    {"role": "Software Engineer", "seniority": "Senior", "location": "Bengaluru, India", "min_exp": 5, "max_exp": 8, "min_salary": 1500000, "max_salary": 2500000, "avg_bonus_pct": 10, "avg_equity_pct": 10},
    {"role": "Software Engineer", "seniority": "Lead/Principal", "location": "Bengaluru, India", "min_exp": 9, "max_exp": 99, "min_salary": 2800000, "max_salary": 4500000, "avg_bonus_pct": 12, "avg_equity_pct": 15},
    
    # Data Scientist - Bengaluru
    {"role": "Data Scientist", "seniority": "Junior", "location": "Bengaluru, India", "min_exp": 0, "max_exp": 1, "min_salary": 500000, "max_salary": 750000, "avg_bonus_pct": 6, "avg_equity_pct": 0},
    {"role": "Data Scientist", "seniority": "Mid", "location": "Bengaluru, India", "min_exp": 2, "max_exp": 4, "min_salary": 1000000, "max_salary": 1600000, "avg_bonus_pct": 9, "avg_equity_pct": 7},
    {"role": "Data Scientist", "seniority": "Senior", "location": "Bengaluru, India", "min_exp": 5, "max_exp": 8, "min_salary": 1800000, "max_salary": 3000000, "avg_bonus_pct": 11, "avg_equity_pct": 12},

    # HR Manager - Bengaluru
    {"role": "HR Manager", "seniority": "Mid", "location": "Bengaluru, India", "min_exp": 3, "max_exp": 6, "min_salary": 700000, "max_salary": 1200000, "avg_bonus_pct": 7, "avg_equity_pct": 0},
    {"role": "HR Manager", "seniority": "Senior", "location": "Bengaluru, India", "min_exp": 7, "max_exp": 12, "min_salary": 1300000, "max_salary": 2000000, "avg_bonus_pct": 10, "avg_equity_pct": 5},
    {"role": "HR Manager", "seniority": "Lead/Principal", "location": "Bengaluru, India", "min_exp": 13, "max_exp": 99, "min_salary": 2100000, "max_salary": 3500000, "avg_bonus_pct": 15, "avg_equity_pct": 8},

    # Business Analyst - Bengaluru
    {"role": "Business Analyst", "seniority": "Junior", "location": "Bengaluru, India", "min_exp": 0, "max_exp": 2, "min_salary": 450000, "max_salary": 700000, "avg_bonus_pct": 5, "avg_equity_pct": 0},
    {"role": "Business Analyst", "seniority": "Mid", "location": "Bengaluru, India", "min_exp": 3, "max_exp": 6, "min_salary": 800000, "max_salary": 1300000, "avg_bonus_pct": 8, "avg_equity_pct": 3},

    # Software Engineer - Mumbai
    {"role": "Software Engineer", "seniority": "Junior", "location": "Mumbai, India", "min_exp": 0, "max_exp": 1, "min_salary": 350000, "max_salary": 550000, "avg_bonus_pct": 5, "avg_equity_pct": 0},
    {"role": "Software Engineer", "seniority": "Mid", "location": "Mumbai, India", "min_exp": 2, "max_exp": 4, "min_salary": 700000, "max_salary": 1100000, "avg_bonus_pct": 8, "avg_equity_pct": 4},

    # Data Scientist - Hyderabad
    {"role": "Data Scientist", "seniority": "Junior", "location": "Hyderabad, India", "min_exp": 0, "max_exp": 1, "min_salary": 450000, "max_salary": 650000, "avg_bonus_pct": 6, "avg_equity_pct": 0},
    {"role": "Data Scientist", "seniority": "Mid", "location": "Hyderabad, India", "min_exp": 2, "max_exp": 4, "min_salary": 900000, "max_salary": 1400000, "avg_bonus_pct": 9, "avg_equity_pct": 6},

    # Product Manager - Bengaluru
    {"role": "Product Manager", "seniority": "Mid", "location": "Bengaluru, India", "min_exp": 4, "max_exp": 7, "min_salary": 1600000, "max_salary": 2500000, "avg_bonus_pct": 12, "avg_equity_pct": 10},
    {"role": "Product Manager", "seniority": "Senior", "location": "Bengaluru, India", "min_exp": 8, "max_exp": 12, "min_salary": 2800000, "max_salary": 4000000, "avg_bonus_pct": 15, "avg_equity_pct": 18},

    # Marketing Specialist - Delhi
    {"role": "Marketing Specialist", "seniority": "Junior", "location": "Delhi, India", "min_exp": 0, "max_exp": 2, "min_salary": 300000, "max_salary": 500000, "avg_bonus_pct": 4, "avg_equity_pct": 0},
    {"role": "Marketing Specialist", "seniority": "Mid", "location": "Delhi, India", "min_exp": 3, "max_exp": 6, "min_salary": 600000, "max_salary": 1000000, "avg_bonus_pct": 7, "avg_equity_pct": 0},
]

# Convert to DataFrame for easier querying
MOCK_SALARY_DF = pd.DataFrame(MOCK_SALARY_DATA)

# --- Advanced Tools Page Function ---
def advanced_tools_page(app_id, FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL):
    user_email = st.session_state.get('user_email', 'anonymous')
    log_user_action(user_email, "ADVANCED_TOOLS_PAGE_ACCESSED")

    # Access dark_mode from session state, defaulting to False if not set
    dark_mode = st.session_state.get('dark_mode_main', False)

    st.markdown(f"""
    <style>
    .advanced-tools-container {{
        background-color: {'#2D2D2D' if dark_mode else 'rgba(255, 255, 255, 0.96)'};
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0px 8px 20px rgba(0,0,0,{'0.2' if dark_mode else '0.1'});
        animation: fadeIn 0.8s ease-in-out;
        color: {'#E0E0E0' if dark_mode else '#333333'};
        margin-bottom: 2rem;
    }}
    .advanced-tools-header {{
        font-size: 2.2rem;
        font-weight: 700;
        color: {'#00cec9' if dark_mode else '#00cec9'};
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #00cec9;
        display: inline-block;
        margin-bottom: 1.5rem;
    }}
    .advanced-tools-caption {{
        font-size: 1.1em;
        color: {'#BBBBBB' if dark_mode else '#555555'};
        margin-bottom: 2rem;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 24px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: nowrap;
        border-radius: 8px 8px 0 0;
        gap: 10px;
        padding-top: 10px;
        padding-bottom: 10px;
        background-color: {'#3A3A3A' if dark_mode else '#f0f2f6'};
        color: {'#BBBBBB' if dark_mode else '#555555'};
        font-weight: 600;
        transition: all 0.2s ease;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {'#00cec9' if dark_mode else '#00cec9'};
        color: white;
        border-bottom: 4px solid {'#00cec9' if dark_mode else '#00cec9'};
    }}
    .stTabs [aria-selected="true"] > div {{
        color: white !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: {'#4A4A4A' if dark_mode else '#e0e2e6'};
    }}
    .stExpander {{
        background-color: {'#3A3A3A' if dark_mode else '#f0f2f6'};
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,{'0.2' if dark_mode else '0.05'});
    }}
    .stExpander > div > div > div > p {{
        color: {'#E0E0E0' if dark_mode else '#333333'};
    }}
    .stExpander > div[data-testid="stExpanderToggle"] {{
        color: {'#00cec9' if dark_mode else '#00cec9'};
    }}
    .stExpander > div[data-testid="stExpanderToggle"] svg {{
        fill: {'#00cec9' if dark_mode else '#00cec9'};
    }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="advanced-tools-container">', unsafe_allow_html=True)
    st.markdown('<div class="advanced-tools-header">📈 Advanced Tools</div>', unsafe_allow_html=True)
    st.markdown('<p class="advanced-tools-caption">Explore powerful HR analytics and automation tools to gain deeper insights and streamline your processes.</p>', unsafe_allow_html=True)

    tab_predictive, tab_skill_gap, tab_compensation, tab_dei, tab_scheduling = st.tabs([ # Added tab_scheduling
        "🔮 Predictive Analytics", "🧩 Skill Gap Analysis", "💰 Compensation Benchmarking", "🤝 DEI Analytics", "🗓️ Interview Scheduling"
    ])

    with tab_predictive:
        st.subheader("Candidate Success Prediction (Mock)")
        st.info("This tool predicts the likelihood of a candidate succeeding in a role based on various factors. (Mock functionality)")

        with st.form("predictive_form", clear_on_submit=False):
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                candidate_score = st.slider("Candidate Score (%)", 0, 100, 75, key="pred_score")
                years_experience = st.slider("Years of Experience", 0.0, 30.0, 5.0, step=0.5, key="pred_exp")
                skills_match_score = st.slider("Specific Skills Match (0-100%)", 0, 100, 70, key="pred_skills_match")
            with col_p2:
                education_level = st.selectbox("Highest Education Level", ["High School", "Associate's", "Bachelor's", "Master's", "PhD"], key="pred_edu")
                interview_feedback = st.slider("Interview Feedback (1-5, 5=Excellent)", 1, 5, 3, key="pred_feedback")
                past_company_tier = st.selectbox("Past Company Tier", ["Tier 1 (FAANG/Unicorn)", "Tier 2 (Large Enterprise)", "Tier 3 (Mid-size)", "Tier 4 (Startup/Small)"], key="pred_company_tier")
            
            predict_button = st.form_submit_button("Predict Success")

            if predict_button:
                # More complex mock prediction logic
                likelihood_score = 0
                if candidate_score >= 80: likelihood_score += 3
                elif candidate_score >= 60: likelihood_score += 2
                else: likelihood_score += 1

                if years_experience >= 5: likelihood_score += 3
                elif years_experience >= 2: likelihood_score += 2
                else: likelihood_score += 1

                if skills_match_score >= 80: likelihood_score += 2
                elif skills_match_score >= 50: likelihood_score += 1

                if interview_feedback >= 4: likelihood_score += 2
                elif interview_feedback >= 3: likelihood_score += 1

                if past_company_tier == "Tier 1 (FAANG/Unicorn)": likelihood_score += 2
                elif past_company_tier == "Tier 2 (Large Enterprise)": likelihood_score += 1

                probability = min(100, max(0, int(likelihood_score / 13 * 100) + np.random.randint(-10, 10))) # Add some randomness
                
                likelihood = "Low"
                confidence = "Low"
                if probability >= 80:
                    likelihood = "High"
                    confidence = "High"
                elif probability >= 50:
                    likelihood = "Moderate"
                    confidence = "Medium"
                else:
                    likelihood = "Low"
                    confidence = "Low"
                
                st.success(f"**Prediction:** The candidate has a **{likelihood}** likelihood of success in this role (Simulated Probability: **{probability}%**).")
                st.info(f"Confidence in this prediction: **{confidence}**.")
                st.write(f"*(Simulated Score: {likelihood_score}/13)*")
                log_user_action(user_email, "CANDIDATE_PREDICTION_USED", {"score": candidate_score, "exp": years_experience, "probability": probability})

        st.markdown("---")
        st.subheader("Employee Churn Prediction (Mock)")
        st.info("Predict which existing employees might be at risk of leaving the company. (Mock functionality)")

        with st.form("churn_prediction_form", clear_on_submit=False):
            col_ch1, col_ch2 = st.columns(2)
            with col_ch1:
                employee_tenure = st.slider("Employee Tenure (Years)", 0, 20, 3, key="churn_tenure")
                performance_rating = st.slider("Last Performance Rating (1-5)", 1, 5, 3, key="churn_perf")
            with col_ch2:
                compensation_satisfaction = st.slider("Compensation Satisfaction (1-5)", 1, 5, 3, key="churn_comp_sat")
                work_life_balance = st.slider("Work-Life Balance (1-5)", 1, 5, 3, key="churn_wlb")
            
            predict_churn_button = st.form_submit_button("Predict Churn Risk")

            if predict_churn_button:
                churn_risk_score = 0
                if employee_tenure <= 2: churn_risk_score += 2 # Early churn risk
                elif employee_tenure >= 7: churn_risk_score += 1 # Long-term stagnation risk

                if performance_rating <= 2: churn_risk_score += 3 # Low performance
                
                if compensation_satisfaction <= 2: churn_risk_score += 3 # Unhappy with pay
                elif compensation_satisfaction == 3: churn_risk_score += 1

                if work_life_balance <= 2: churn_risk_score += 2 # Poor WLB

                risk_level = "Low"
                if churn_risk_score >= 6:
                    risk_level = "High"
                elif churn_risk_score >= 3:
                    risk_level = "Moderate"
                
                st.warning(f"**Churn Risk Prediction:** This employee has a **{risk_level}** risk of leaving. (Simulated Score: {churn_risk_score}/11)")
                st.write("*(Factors considered: Tenure, Performance, Compensation Satisfaction, Work-Life Balance)*")
                log_user_action(user_email, "CHURN_PREDICTION_USED", {"tenure": employee_tenure, "risk": risk_level})


    with tab_skill_gap:
        st.subheader("Skill Gap Analysis (Mock)")
        st.info("Identify common and missing skills by comparing required skills for a role against a candidate's profile. (Mock functionality)")

        with st.form("skill_gap_form", clear_on_submit=False):
            required_skills_input = st.text_area("Required Skills (comma-separated)", "Python, SQL, Machine Learning, Communication, Problem Solving", height=100, key="req_skills")
            candidate_skills_input = st.text_area("Candidate's Skills (comma-separated)", "Python, SQL, Data Analysis, Teamwork", height=100, key="cand_skills")
            
            analyze_button = st.form_submit_button("Analyze Skill Gap")

            if analyze_button:
                req_skills = set([s.strip().lower() for s in required_skills_input.split(',') if s.strip()])
                cand_skills = set([s.strip().lower() for s in candidate_skills_input.split(',') if s.strip()])

                matched_skills = req_skills.intersection(cand_skills)
                missing_skills = req_skills.difference(cand_skills)
                additional_skills = cand_skills.difference(req_skills)

                st.markdown("#### Analysis Results:")
                if matched_skills:
                    st.success(f"✅ **Matched Skills:** {', '.join(matched_skills).title()}")
                else:
                    st.warning("No direct skill matches found.")
                
                if missing_skills:
                    st.error(f"❌ **Missing Skills:** {', '.join(missing_skills).title()}")
                    # New: Suggest learning resources
                    st.markdown("##### Suggested Learning Resources for Missing Skills:")
                    for skill in missing_skills:
                        st.write(f"- For **{skill.title()}**: [Coursera Course](https://www.coursera.org/courses?query={skill.replace(' ', '%20')}), [Udemy Course](https://www.udemy.com/courses/search/?src=ukw&q={skill.replace(' ', '%20')}) (Mock Links)")
                else:
                    st.info("Candidate possesses all required skills!")

                if additional_skills:
                    st.info(f"💡 **Additional Skills (not required but present):** {', '.join(additional_skills).title()}")

                log_user_action(user_email, "SKILL_GAP_ANALYSI_USED", {"required": required_skills_input, "candidate": candidate_skills_input})
        
        st.markdown("---")
        st.subheader("Role Skill Requirements Builder (Mock)")
        st.info("Define and categorize skills required for a new role. (Mock functionality)")
        
        new_role_name = st.text_input("New Role Name", "Senior AI Engineer", key="new_role_skill_builder")
        core_skills = st.text_area("Core Skills (comma-separated)", "Deep Learning, Python, TensorFlow, PyTorch", key="core_skills_builder")
        soft_skills = st.text_area("Soft Skills (comma-separated)", "Communication, Teamwork, Problem Solving", key="soft_skills_builder")
        
        if st.button("Build Skill Profile", key="build_skill_profile_button"):
            st.success(f"Skill profile built for **{new_role_name}**!")
            st.write(f"**Core Skills:** {core_skills}")
            st.write(f"**Soft Skills:** {soft_skills}")
            st.info("This profile can be saved and used for future candidate matching.")
            log_user_action(user_email, "ROLE_SKILL_BUILDER_USED", {"role": new_role_name})

        st.markdown("---")
        st.subheader("Team Skill Inventory & Heatmap (Mock Visualization)")
        st.info("Visualize the distribution and proficiency of key skills across your current team. (Mock data)")
        
        # Mock Team Skill Data
        team_skills_data = {
            'Skill': ['Python', 'SQL', 'Cloud Computing', 'Project Management', 'Data Analysis', 'Communication', 'Leadership'],
            'Proficiency (Avg)': [4.2, 3.8, 3.0, 4.5, 3.5, 4.8, 4.0],
            'Team Members': [15, 12, 8, 10, 14, 20, 7]
        }
        team_skills_df = pd.DataFrame(team_skills_data)

        fig_team_skills = px.bar(
            team_skills_df.sort_values('Proficiency (Avg)', ascending=False),
            x='Proficiency (Avg)',
            y='Skill',
            orientation='h',
            title='Average Team Proficiency by Skill',
            labels={'Proficiency (Avg)': 'Average Proficiency (1-5)', 'Skill': 'Skill'},
            color='Proficiency (Avg)',
            color_continuous_scale=px.colors.sequential.Teal if not dark_mode else px.colors.sequential.Plasma
        )
        st.plotly_chart(fig_team_skills, use_container_width=True)
        st.caption("This chart shows average self-reported or assessed proficiency for key skills across the team.")

        # Mock Team Skill Heatmap
        st.markdown("##### Team Skill Heatmap (Simulated)")
        skills_for_heatmap = ['Python', 'SQL', 'Cloud Computing', 'Communication', 'Leadership']
        team_members_for_heatmap = [f"Team Member {i+1}" for i in range(10)]
        
        heatmap_data = pd.DataFrame(
            np.random.randint(1, 6, size=(len(team_members_for_heatmap), len(skills_for_heatmap))),
            index=team_members_for_heatmap,
            columns=skills_for_heatmap
        )
        
        fig_heatmap = px.imshow(
            heatmap_data,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=px.colors.sequential.Greens if not dark_mode else px.colors.sequential.Viridis,
            title="Team Skill Proficiency Heatmap (1=Low, 5=High)"
        )
        fig_heatmap.update_xaxes(side="top")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.caption("A visual representation of individual skill strengths across the team.")


    with tab_compensation:
        st.subheader("💰 Compensation Benchmarking")
        st.info("Get estimated salary ranges based on role, experience, seniority, and location. Data is simulated for demonstration.")

        # Get unique roles, seniorities, and locations from mock data
        all_roles = sorted(MOCK_SALARY_DF['role'].unique().tolist())
        all_seniorities = sorted(MOCK_SALARY_DF['seniority'].unique().tolist(), key=lambda x: ['Junior', 'Mid', 'Senior', 'Lead/Principal'].index(x))
        all_locations = sorted(MOCK_SALARY_DF['location'].unique().tolist())

        # Try to infer roles/locations from comprehensive_df if available in session state
        if 'comprehensive_df' in st.session_state and not st.session_state['comprehensive_df'].empty:
            df_comp = st.session_state['comprehensive_df']
            if 'Desired Role' in df_comp.columns:
                inferred_roles = df_comp['Desired Role'].dropna().unique().tolist()
                all_roles = sorted(list(set(all_roles + inferred_roles)))
            if 'Location' in df_comp.columns:
                inferred_locations = df_comp['Location'].dropna().unique().tolist()
                # Clean inferred locations (e.g., split by comma if multiple, remove 'not found')
                cleaned_inferred_locations = []
                for loc_str in inferred_locations:
                    cleaned_inferred_locations.extend([loc.strip() for loc in loc_str.split(',') if loc.strip() and loc.strip().lower() != 'not found'])
                all_locations = sorted(list(set(all_locations + cleaned_inferred_locations)))
            st.caption("💡 Roles and locations are pre-populated with common values and inferred from your screened candidates.")

        with st.form("compensation_form", clear_on_submit=False):
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                selected_role = st.selectbox("Role Title", all_roles, key="comp_role")
                selected_seniority = st.selectbox("Seniority Level", all_seniorities, key="comp_seniority")
            with col_c2:
                years_exp_comp = st.slider("Years of Experience", 0, 20, 5, key="comp_exp")
                selected_location = st.selectbox("Location", all_locations, key="comp_loc")
            
            benchmark_button = st.form_submit_button("Get Benchmark")

            if benchmark_button:
                # Filter mock data based on user selection
                filtered_salaries = MOCK_SALARY_DF[
                    (MOCK_SALARY_DF['role'] == selected_role) &
                    (MOCK_SALARY_DF['seniority'] == selected_seniority) &
                    (MOCK_SALARY_DF['location'] == selected_location) &
                    (MOCK_SALARY_DF['min_exp'] <= years_exp_comp) &
                    (MOCK_SALARY_DF['max_exp'] >= years_exp_comp)
                ]

                if not filtered_salaries.empty:
                    # Get the specific benchmark row
                    benchmark_row = filtered_salaries.iloc[0] # Take the first match

                    base_min_orig = benchmark_row['min_salary']
                    base_max_orig = benchmark_row['max_salary']
                    avg_bonus_pct = benchmark_row['avg_bonus_pct']
                    avg_equity_pct = benchmark_row['avg_equity_pct']

                    # Calculate total compensation (simulated)
                    avg_base_salary_orig = (base_min_orig + base_max_orig) / 2
                    
                    total_comp_min_orig = base_min_orig + (base_min_orig * avg_bonus_pct / 100) + (base_min_orig * avg_equity_pct / 100)
                    total_comp_max_orig = base_max_orig + (base_max_orig * avg_bonus_pct / 100) + (base_max_orig * avg_equity_pct / 100)

                    st.markdown("#### Benchmark Results:")
                    st.success(f"**Estimated Compensation Range for '{selected_seniority} {selected_role}' in '{selected_location}' ({years_exp_comp} yrs exp):**")
                    st.write(f"**Base Salary: ₹{base_min_orig:,.0f} - ₹{base_max_orig:,.0f} per annum**")
                    st.write(f"**Total Compensation (incl. Bonus/Equity): ₹{total_comp_min_orig:,.0f} - ₹{total_comp_max_orig:,.0f} per annum (approx.)**")

                    # Visualization of the range
                    salary_range_df = pd.DataFrame({
                        'Component': ['Base Min', 'Base Max', 'Total Comp Min', 'Total Comp Max'],
                        'Salary': [base_min_orig, base_max_orig, total_comp_min_orig, total_comp_max_orig]
                    })
                    fig_salary_range = px.bar(
                        salary_range_df,
                        x='Component',
                        y='Salary',
                        title='Estimated Compensation Breakdown',
                        labels={'Salary': 'Annual Salary (₹)'},
                        color_discrete_sequence=['#00cec9', '#00b0a8', '#2ecc71', '#27ae60'] if not dark_mode else ['#00cec9', '#00b0a8', '#2ecc71', '#27ae60']
                    )
                    st.plotly_chart(fig_salary_range, use_container_width=True)

                    st.markdown("---")
                    st.markdown("##### Key Factors & Confidence (Simulated):")
                    st.write(f"- **Experience Level:** {years_exp_comp} years (matched to {benchmark_row['min_exp']}-{benchmark_row['max_exp']} yrs band)")
                    st.write(f"- **Seniority:** {selected_seniority}")
                    st.write(f"- **Location Premium:** {selected_location}")
                    st.write(f"- **Average Bonus:** {avg_bonus_pct}%")
                    st.write(f"- **Average Equity/Stock:** {avg_equity_pct}%")
                    st.info("Confidence Level: **High** (direct match found in simulated data)")

                    # New: Custom Range Adjustment
                    st.markdown("---")
                    st.subheader("Custom Range Adjustment")
                    st.write("Adjust the benchmarked range to fit specific budget or candidate considerations.")
                    col_adj1, col_adj2 = st.columns(2)
                    with col_adj1:
                        adjusted_min_salary = st.number_input("Adjusted Minimum Base Salary (₹)", min_value=0, value=int(base_min_orig), step=50000, key="adj_min_salary")
                    with col_adj2:
                        adjusted_max_salary = st.number_input("Adjusted Maximum Base Salary (₹)", min_value=0, value=int(base_max_orig), step=50000, key="adj_max_salary")
                    
                    if st.button("Apply Adjustment", key="apply_adj_button"):
                        st.success(f"Adjusted Base Salary Range: ₹{adjusted_min_salary:,.0f} - ₹{adjusted_max_salary:,.0f} per annum.")
                        log_user_action(user_email, "COMPENSATION_ADJUSTED", {"min": adjusted_min_salary, "max": adjusted_max_salary})

                    # New: Scenario Analysis
                    st.markdown("---")
                    st.subheader("Compensation Scenario Analysis (Mock)")
                    st.write("See how different bonus or equity percentages impact the total compensation.")
                    scenario_bonus_pct = st.slider("Scenario Bonus (%)", 0, 30, avg_bonus_pct, key="scenario_bonus")
                    scenario_equity_pct = st.slider("Scenario Equity (%)", 0, 30, avg_equity_pct, key="scenario_equity")

                    if st.button("Run Scenario", key="run_scenario_button"):
                        scenario_total_comp_min = base_min_orig + (base_min_orig * scenario_bonus_pct / 100) + (base_min_orig * scenario_equity_pct / 100)
                        scenario_total_comp_max = base_max_orig + (base_max_orig * scenario_bonus_pct / 100) + (base_max_orig * scenario_equity_pct / 100)
                        st.info(f"Scenario Total Compensation: ₹{scenario_total_comp_min:,.0f} - ₹{scenario_total_comp_max:,.0f} per annum.")
                        log_user_action(user_email, "COMPENSATION_SCENARIO_RUN", {"bonus": scenario_bonus_pct, "equity": scenario_equity_pct})


                    # Compare with Candidate's Expected Salary
                    st.markdown("---")
                    st.subheader("Compare with Candidate's Expected Salary")
                    candidate_expected_salary = st.number_input("Candidate's Expected Annual Salary (₹)", min_value=0, value=int(avg_base_salary_orig), step=10000, key="cand_expected_salary")
                    
                    if st.button("Compare", key="compare_salary_button"):
                        if candidate_expected_salary < base_min_orig:
                            st.warning(f"Candidate's expectation (₹{candidate_expected_salary:,.0f}) is below the benchmark range (₹{base_min_orig:,.0f} - ₹{base_max_orig:,.0f}).")
                            st.write("This could be an opportunity to offer a competitive package or indicate a mismatch in expectations.")
                        elif candidate_expected_salary > base_max_orig:
                            st.warning(f"Candidate's expectation (₹{candidate_expected_salary:,.0f}) is above the benchmark range (₹{base_min_orig:,.0f} - ₹{base_max_orig:,.0f}).")
                            st.write("This might require re-evaluation of the role's budget or negotiation strategy.")
                        else:
                            st.success(f"Candidate's expectation (₹{candidate_expected_salary:,.0f}) is within the benchmark range (₹{base_min_orig:,.0f} - ₹{base_max_orig:,.0f}).")
                            st.write("This indicates a good alignment with market compensation.")
                        log_user_action(user_email, "COMPENSATION_COMPARE_USED", {"expected_salary": candidate_expected_salary, "benchmark_min": base_min_orig, "benchmark_max": base_max_orig})

                else:
                    st.warning("No precise benchmark data found for the selected criteria. Adjust parameters or consider a broader search.")
                    st.info("Confidence Level: **Low** (no direct match in simulated data)")
                
                log_user_action(user_email, "COMPENSATION_BENCHMARK_USED", {
                    "role": selected_role, "seniority": selected_seniority, "location": selected_location, "exp": years_exp_comp,
                    "benchmark_found": not filtered_salaries.empty
                })

    with tab_dei:
        st.subheader("Diversity, Equity, and Inclusion (DEI) Analytics (Mock)")
        st.info("Visualize and analyze diversity metrics within your candidate pipeline or workforce. (Mock functionality)")

        st.write("#### Candidate Gender Distribution (Mock Data)")
        # Mock data for gender distribution
        gender_data = pd.DataFrame({
            'Gender': ['Male', 'Female', 'Non-binary', 'Prefer not to say'],
            'Count': [np.random.randint(50, 150), np.random.randint(40, 120), np.random.randint(5, 20), np.random.randint(10, 30)]
        })

        fig_gender = px.pie(
            gender_data,
            values='Count',
            names='Gender',
            title='Applicant Gender Breakdown',
            color_discrete_sequence=px.colors.qualitative.Pastel if not dark_mode else px.colors.qualitative.Dark2
        )
        st.plotly_chart(fig_gender, use_container_width=True)

        st.write("#### Candidate Age Group Distribution (Mock Data)")
        # Mock data for age group distribution
        age_data = pd.DataFrame({
            'Age Group': ['18-24', '25-34', '35-44', '45-54', '55+'],
            'Count': [np.random.randint(30, 80), np.random.randint(70, 180), np.random.randint(50, 100), np.random.randint(20, 60), np.random.randint(10, 30)]
        })

        fig_age = px.bar(
            age_data,
            x='Age Group',
            y='Count',
            title='Applicant Age Group Distribution',
            color='Count',
            color_continuous_scale=px.colors.sequential.Viridis if not dark_mode else px.colors.sequential.Cividis
        )
        st.plotly_chart(fig_age, use_container_width=True)

        st.write("#### Diversity by Department (Mock Data)")
        department_diversity_data = pd.DataFrame({
            'Department': ['Engineering', 'Sales', 'HR', 'Marketing', 'Product'],
            'Female Representation (%)': [np.random.uniform(20, 40), np.random.uniform(30, 50), np.random.uniform(50, 70), np.random.uniform(40, 60), np.random.uniform(25, 45)],
            'Underrepresented Groups (%)': [np.random.uniform(10, 25), np.random.uniform(15, 30), np.random.uniform(10, 20), np.random.uniform(12, 28), np.random.uniform(8, 22)],
        })
        fig_dept_diversity = px.bar(
            department_diversity_data,
            x='Department',
            y=['Female Representation (%)', 'Underrepresented Groups (%)'],
            barmode='group',
            title='Diversity Metrics by Department',
            labels={'value': 'Percentage', 'variable': 'Diversity Metric'},
            color_discrete_sequence=['#00cec9', '#fab1a0'] if not dark_mode else ['#00cec9', '#fab1a0']
        )
        st.plotly_chart(fig_dept_diversity, use_container_width=True)

        st.markdown("---")
        st.subheader("Hiring Funnel Diversity Breakdown (Mock)")
        st.info("Analyze diversity metrics at each stage of your hiring pipeline. (Mock data)")

        funnel_data = pd.DataFrame({
            'Stage': ['Applicants', 'Screened', 'Interviewed', 'Offered', 'Hired'],
            'Total': [1000, 500, 100, 20, 10],
            'Female': [400, 200, 40, 8, 4],
            'Underrepresented Groups': [150, 70, 15, 3, 2]
        })
        funnel_data['Female %'] = (funnel_data['Female'] / funnel_data['Total'] * 100).round(1)
        funnel_data['URG %'] = (funnel_data['Underrepresented Groups'] / funnel_data['Total'] * 100).round(1)

        fig_funnel = px.line(
            funnel_data,
            x='Stage',
            y=['Female %', 'URG %'],
            title='Diversity Percentage Across Hiring Funnel',
            labels={'value': 'Percentage (%)', 'variable': 'Diversity Group'},
            markers=True,
            color_discrete_sequence=['#00cec9', '#fab1a0'] if not dark_mode else ['#00cec9', '#fab1a0']
        )
        st.plotly_chart(fig_funnel, use_container_width=True)
        st.dataframe(funnel_data[['Stage', 'Total', 'Female %', 'URG %']], use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Pay Equity Analysis (Mock)")
        st.info("Simulate analysis of pay differences across demographic groups for similar roles. (Mock functionality)")
        
        gender_pay_gap = np.random.uniform(-5, 5) # Simulate a small gender pay gap
        urg_pay_gap = np.random.uniform(-3, 3) # Simulate a small URG pay gap

        if st.button("Run Pay Equity Analysis", key="run_pay_equity_button"):
            st.markdown("##### Simulated Pay Equity Report:")
            st.write(f"- **Gender Pay Gap (Female vs. Male):** **{gender_pay_gap:.2f}%** (simulated, positive means male earns more)")
            st.write(f"- **Underrepresented Groups Pay Gap:** **{urg_pay_gap:.2f}%** (simulated, positive means non-URG earns more)")
            
            if abs(gender_pay_gap) > 2 or abs(urg_pay_gap) > 2:
                st.warning("⚠️ **Action Recommended:** Simulated data indicates potential pay disparities. Further investigation is advised.")
            else:
                st.success("✅ **Simulated Result:** Pay appears generally equitable based on available data.")
            st.caption("_This is a simplified simulation and does not account for all complex factors in real pay equity analysis._")
            log_user_action(user_email, "PAY_EQUITY_ANALYSIS_USED")


        st.markdown("---")
        st.write("More DEI metrics (e.g., ethnicity, disability status) and bias detection features could be added here.")
        
        # New: Mock Bias Detection for Job Description
        st.subheader("Job Description Bias Detection (Mock)")
        st.info("Analyze your job description for potentially biased language. (Mock functionality)")
        jd_text_for_bias = st.text_area("Paste Job Description Text for Bias Check:", "We are seeking a highly motivated and aggressive individual to lead our sales team. Must be a rockstar with a proven track record.", height=150, key="jd_bias_check")
        
        if st.button("Check for Bias", key="check_bias_button"):
            biased_terms = []
            bias_score = 0
            
            # Gender-coded words
            gender_coded = {"aggressive": "masculine", "dominant": "masculine", "leader": "masculine", "competitive": "masculine",
                            "nurturing": "feminine", "supportive": "feminine", "collaborative": "feminine"}
            for term, gender in gender_coded.items():
                if term in jd_text_for_bias.lower():
                    biased_terms.append(f"{term} ({gender}-coded)")
                    bias_score += 1

            # Age/culture/other words
            other_biased = {"rockstar": "can imply age/culture bias", "ninja": "can imply culture bias", "guru": "can imply age bias",
                            "digital native": "age bias", "young": "age bias", "energetic": "age bias"}
            for term, reason in other_biased.items():
                if term in jd_text_for_bias.lower():
                    biased_terms.append(f"{term} ({reason})")
                    bias_score += 1

            if biased_terms:
                st.warning("⚠️ Potential biased language detected:")
                for term in biased_terms:
                    st.write(f"- `{term}`: Consider using more neutral alternatives.")
                st.markdown("Suggested alternatives: 'driven', 'high-achieving', 'expert', 'specialist', 'innovative', 'team-oriented'.")
                st.error(f"**Overall Bias Score (Simulated):** {bias_score} (Higher score indicates more bias)")
            else:
                st.success("✅ No obvious biased language detected in this text. Great job!")
                st.info(f"**Overall Bias Score (Simulated):** {bias_score}")
            log_user_action(user_email, "JD_BIAS_CHECK_USED")

    with tab_scheduling:
        st.subheader("🗓️ Automated Interview Scheduling")
        st.info("Streamline your interview process by automating scheduling, reminders, and feedback collection. Data is stored in Firebase.")
        
        st.markdown("---")
        st.subheader("📧 Email Configuration (Gmail App Password)")
        st.warning("""
            **To send actual emails via Gmail:**
            1.  **Enable 2-Step Verification** for your Google Account.
            2.  Go to [Google Account Security](https://myaccount.google.com/security) -> "App passwords" (you might need to search for it).
            3.  Generate a new App password and **copy the 16-character code**.
            4.  **REPLACE THE PLACEHOLDERS BELOW IN THE CODE WITH YOUR ACTUAL GMAIL ADDRESS AND APP PASSWORD.**
            5.  **Note:** Direct SMTP connections from this Streamlit Canvas environment might still be blocked by network policies. For reliable email sending, consider deploying your app to a server where you have full control over network access.
        """)
        
        # --- HARDCODED GMAIL CREDENTIALS (REPLACE THESE PLACEHOLDERS) ---
        # You need to replace "YOUR_GMAIL_ADDRESS@gmail.com" with your actual Gmail address
        # and "YOUR_GMAIL_APP_PASSWORD" with the 16-character App Password you generated.
        gmail_address = "screenerpro.ai@gmail.com"  # <--- REPLACE THIS
        gmail_app_password = "zdbvrxatqutupql"  # <--- REPLACE THIS
        # --- END HARDCODED GMAIL CREDENTIALS ---

        # Store these in session state for consistency, but they are now hardcoded values
        st.session_state.gmail_address = gmail_address
        st.session_state.gmail_app_password = gmail_app_password

        # Commented out UI input fields as requested
        # st.session_state.gmail_address = st.text_input("Your Gmail Address", value=st.session_state.gmail_address, help="e.g., your.email@gmail.com", key="gmail_address_input")
        # st.session_state.gmail_app_password = st.text_input("Gmail App Password", value=st.session_state.gmail_app_password, type="password", help="Your 16-character App Password from Google Account Security.", key="gmail_app_password_input")
        
        # We don't need a toggle anymore, as the presence of credentials implies attempting to send.
        # send_real_emails_toggle = st.checkbox("Attempt to Send Real Emails (Requires Configuration Above)", key="send_real_emails_toggle")


        # --- Load existing interviews, feedback, and interviewers from Firebase ---
        if 'user_interviews' not in st.session_state:
            st.session_state.user_interviews = []
        if 'user_feedback' not in st.session_state:
            st.session_state.user_feedback = []
        if 'user_interviewers' not in st.session_state:
            st.session_state.user_interviewers = []

        # Load data on page load
        success_interviews, loaded_interviews = load_collection_from_firestore(
            f"artifacts/{app_id}/users/{user_email}/interviews", FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL
        )
        if success_interviews:
            st.session_state.user_interviews = loaded_interviews
        else:
            st.error(f"Failed to load interviews: {loaded_interviews}")

        success_feedback, loaded_feedback = load_collection_from_firestore(
            f"artifacts/{app_id}/users/{user_email}/interview_feedback", FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL
        )
        if success_feedback:
            st.session_state.user_feedback = loaded_feedback
        else:
            st.error(f"Failed to load feedback: {loaded_feedback}")
        
        success_interviewers, loaded_interviewers = load_collection_from_firestore(
            f"artifacts/{app_id}/users/{user_email}/interviewers", FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL
        )
        if success_interviewers:
            st.session_state.user_interviewers = loaded_interviewers
        else:
            st.error(f"Failed to load interviewers: {loaded_interviewers}")

        # --- Manage Interviewers Section ---
        st.markdown("---")
        with st.expander("👤 Manage Interviewers"):
            st.markdown("##### Add New Interviewer")
            with st.form("add_interviewer_form", clear_on_submit=True):
                new_interviewer_name = st.text_input("Interviewer Name", key="new_interviewer_name_input")
                new_interviewer_email = st.text_input("Interviewer Email", help="This email will receive interview invites.", key="new_interviewer_email_input")
                new_interviewer_general_availability = st.text_input("General Availability (e.g., Mon-Fri 9 AM - 5 PM)", key="new_interviewer_availability_input")
                
                add_interviewer_button = st.form_submit_button("Add Interviewer")

                if add_interviewer_button:
                    if new_interviewer_name and new_interviewer_email:
                        interviewer_data = {
                            "name": new_interviewer_name,
                            "email": new_interviewer_email,
                            "general_availability": new_interviewer_general_availability,
                            "timestamp": datetime.now()
                        }
                        # Use interviewer email as doc ID for easy lookup
                        # Replace . and @ for valid Firebase document ID
                        doc_id = new_interviewer_email.replace('.', '_').replace('@', '_') 
                        success, response = save_document_to_firestore(
                            f"artifacts/{app_id}/users/{user_email}/interviewers", 
                            doc_id,
                            interviewer_data, FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL
                        )
                        if success:
                            st.success(f"Interviewer '{new_interviewer_name}' added successfully to Firebase!")
                            log_user_action(user_email, "INTERVIEWER_ADDED", {"name": new_interviewer_name})
                            st.rerun() # Rerun to refresh the list
                        else:
                            st.error(f"Failed to add interviewer: {response}")
                            log_user_action(user_email, "INTERVIEWER_ADD_FAILED", {"name": new_interviewer_name, "error": response})
                    else:
                        st.warning("Please provide interviewer name and email.")
            
            st.markdown("##### Existing Interviewers")
            if st.session_state.user_interviewers:
                interviewer_df = pd.DataFrame(st.session_state.user_interviewers)
                st.dataframe(interviewer_df[['name', 'email', 'general_availability']], use_container_width=True, hide_index=True)
            else:
                st.info("No interviewers added yet.")

        # --- Schedule a New Interview Form ---
        st.markdown("---")
        with st.form("interview_scheduling_form", clear_on_submit=True):
            st.markdown("##### Schedule a New Interview")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                candidate_name = st.text_input("Candidate Name", key="sched_cand_name")
                candidate_email = st.text_input("Candidate Email", key="sched_cand_email")
                interview_type = st.selectbox("Interview Type", ["Initial Screen", "Technical Interview (Round 1)", "Technical Interview (Round 2)", "Hiring Manager Interview", "Final Round"], key="sched_interview_type")
            
            with col_s2:
                # Dynamically populate interviewer selection
                interviewer_options = ["Select Interviewer"] + [i['name'] for i in st.session_state.user_interviewers]
                selected_interviewer_name = st.selectbox("Select Interviewer", interviewer_options, key="sched_interviewer_name_select")

                selected_interviewer_email = ""
                if selected_interviewer_name != "Select Interviewer":
                    # Find the email for the selected interviewer
                    for interviewer in st.session_state.user_interviewers:
                        if interviewer['name'] == selected_interviewer_name:
                            selected_interviewer_email = interviewer['email']
                            break
                    st.text_input("Interviewer Email (Auto-filled)", value=selected_interviewer_email, disabled=True, key="sched_interviewer_email_display")
                else:
                    st.text_input("Interviewer Email (Auto-filled)", value="", disabled=True, key="sched_interviewer_email_display_empty")

                interview_date = st.date_input("Preferred Date", min_value=datetime.now().date(), key="sched_date")
                interview_time = st.time_input("Preferred Time", value=datetime.now().time(), step=timedelta(minutes=30), key="sched_time")
            
            interview_duration = st.slider("Interview Duration (minutes)", 30, 120, 60, step=15, key="sched_duration")
            interview_notes = st.text_area("Internal Notes for Interviewers", height=80, key="sched_notes")

            schedule_button = st.form_submit_button("Schedule Interview")

            if schedule_button:
                if not candidate_name or not candidate_email or selected_interviewer_name == "Select Interviewer" or not selected_interviewer_email:
                    st.error("Please fill in all required fields (Candidate Name/Email, and select/add an Interviewer).")
                else:
                    interview_data = {
                        "candidate_name": candidate_name,
                        "candidate_email": candidate_email,
                        "interview_type": interview_type,
                        "interviewer_name": selected_interviewer_name,
                        "interviewer_email": selected_interviewer_email,
                        "interview_datetime": datetime.combine(interview_date, interview_time),
                        "duration_minutes": interview_duration,
                        "notes": interview_notes,
                        "scheduled_by": user_email,
                        "timestamp": datetime.now()
                    }
                    
                    success, response = add_document_to_firestore_collection(
                        f"artifacts/{app_id}/users/{user_email}/interviews", interview_data, FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL
                    )

                    if success:
                        st.success(f"✅ Interview scheduled for {candidate_name} with {selected_interviewer_name} on {interview_date} at {interview_time} for {interview_duration} minutes ({interview_type}). (Data saved to Firebase)")
                        st.write("---")
                        st.markdown("##### Notification Status:")
                        
                        # Prepare email content
                        candidate_subject = f"Interview Invitation: {interview_type} with {selected_interviewer_name}"
                        candidate_body = f"""
Dear {candidate_name},

We are pleased to invite you for an interview for the {interview_type} round.

Interviewer: {selected_interviewer_name}
Date: {interview_date.strftime('%Y-%m-%d')}
Time: {interview_time.strftime('%I:%M %p')}
Duration: {interview_duration} minutes

We look forward to speaking with you!

Best regards,
The HR Team
"""
                        interviewer_subject = f"Interview Scheduled: {candidate_name} ({interview_type})"
                        interviewer_body = f"""
Dear {selected_interviewer_name},

An interview has been scheduled for you:

Candidate: {candidate_name}
Candidate Email: {candidate_email}
Interview Type: {interview_type}
Date: {interview_date.strftime('%Y-%m-%d')}
Time: {interview_time.strftime('%I:%M %p')}
Duration: {interview_duration} minutes
Internal Notes: {interview_notes if interview_notes else 'N/A'}

Please add this to your calendar.

Best regards,
The HR Team
"""

                        # Attempt to send real emails if hardcoded credentials are provided
                        if st.session_state.gmail_address and st.session_state.gmail_app_password:
                            st.info("Attempting to send real emails via Gmail (using hardcoded credentials)...")
                            # Send email to candidate
                            send_actual_email(
                                candidate_email,
                                candidate_subject,
                                candidate_body,
                                st.session_state.gmail_address,
                                st.session_state.gmail_app_password
                            )
                            # Send email to interviewer
                            send_actual_email(
                                selected_interviewer_email,
                                interviewer_subject,
                                interviewer_body,
                                st.session_state.gmail_address,
                                st.session_state.gmail_app_password
                            )
                        else:
                            st.info("Hardcoded Gmail credentials not provided. Emails will not be sent.")
                            st.info(f"📧 **Simulated Email to Candidate ({candidate_email}):** Your interview for {interview_type} is scheduled for {interview_date.strftime('%Y-%m-%d')} at {interview_time.strftime('%I:%M %p')}.")
                            st.info(f"📧 **Simulated Calendar Invite to Interviewer ({selected_interviewer_email}):** Interview for {candidate_name} on {interview_date.strftime('%Y-%m-%d')} at {interview_time.strftime('%I:%M %p')}.")
                        
                        st.success("Simulated reminders will be sent automatically 24 hours prior. (Mock)")
                        log_user_action(user_email, "INTERVIEW_SCHEDULED_FIREBASE", {"candidate": candidate_name, "interviewer": selected_interviewer_name, "type": interview_type})
                        st.rerun() # Rerun to refresh the upcoming interviews list
                    else:
                        st.error(f"❌ Failed to save interview to Firebase: {response}")
                        log_user_action(user_email, "INTERVIEW_SCHEDULE_FAILED_FIREBASE", {"candidate": candidate_name, "error": response})

        st.markdown("---")
        st.subheader("Interviewer Availability (Mock)")
        st.info("View mock availability for interviewers to help with manual scheduling. (Mock data)")
        
        # Populate interviewer dropdown from Firebase
        interviewer_avail_options = ["Select Interviewer"] + [i['name'] for i in st.session_state.user_interviewers]
        selected_avail_interviewer = st.selectbox("Select Interviewer", interviewer_avail_options, key="interviewer_avail_select")
        check_date = st.date_input("Check Availability for Date", min_value=datetime.now().date(), key="avail_check_date")

        if st.button("Check Availability", key="check_avail_button"):
            if selected_avail_interviewer == "Select Interviewer":
                st.warning("Please select an interviewer to check availability.")
            else:
                interviewer_general_avail = "N/A"
                for interviewer in st.session_state.user_interviewers:
                    if interviewer['name'] == selected_avail_interviewer:
                        interviewer_general_avail = interviewer.get('general_availability', 'N/A')
                        break

                st.markdown(f"##### Mock Availability for {selected_avail_interviewer} on {check_date}:")
                st.write(f"**General Availability:** {interviewer_general_avail}")
                # Simulate specific availability based on general availability
                if "Mon-Fri 9 AM - 5 PM" in interviewer_general_avail:
                    st.success("✅ Available: 10:00 AM - 12:00 PM, 02:00 PM - 04:00 PM")
                elif "Flexible" in interviewer_general_avail:
                    st.info("⚠️ Flexible availability, please confirm directly.")
                else:
                    st.warning("⚠️ Limited Availability: Please contact directly for specific times.")
                log_user_action(user_email, "INTERVIEWER_AVAILABILITY_CHECKED", {"interviewer": selected_avail_interviewer, "date": check_date})

        st.markdown("---")
        st.subheader("Automated Reminders Configuration (Mock)")
        st.info("Configure automated email reminders for candidates and interviewers. (Mock functionality)")
        
        reminder_candidate_days = st.slider("Send Candidate Reminder (days before interview)", 0, 3, 1, key="rem_cand_days")
        reminder_interviewer_hours = st.slider("Send Interviewer Reminder (hours before interview)", 0, 48, 24, key="rem_int_hours")
        
        if st.button("Save Reminder Settings", key="save_reminders_button"):
            st.success(f"Reminder settings saved: Candidate {reminder_candidate_days} day(s) before, Interviewer {reminder_interviewer_hours} hour(s) before.")
            log_user_action(user_email, "REMINDER_SETTINGS_SAVED")


        st.markdown("---")
        st.subheader("Upcoming Interviews")
        st.info("View your upcoming interview schedule from Firebase.")

        if st.session_state.user_interviews:
            # Sort interviews by date/time
            sorted_interviews = sorted(st.session_state.user_interviews, key=lambda x: x.get('interview_datetime', datetime.min))
            
            display_interviews = []
            for interview in sorted_interviews:
                display_interviews.append({
                    "Candidate": interview.get('candidate_name', 'N/A'),
                    "Role": interview.get('interview_type', 'N/A'), # Reusing type as role for display simplicity
                    "Interviewer": interview.get('interviewer_name', 'N/A'),
                    "Date": interview.get('interview_datetime', datetime.min).strftime("%Y-%m-%d"),
                    "Time": interview.get('interview_datetime', datetime.min).strftime("%I:%M %p")
                })
            st.dataframe(pd.DataFrame(display_interviews), use_container_width=True, hide_index=True)
        else:
            st.info("No upcoming interviews scheduled yet.")

        st.markdown("---")
        st.subheader("Interview Feedback Collection & Trends")
        st.info("Submit and review interview feedback easily, and see overall trends. Data is stored in Firebase.")

        # Populate candidate dropdown from scheduled interviews
        candidate_options = ["New Candidate..."]
        if st.session_state.user_interviews:
            candidate_options.extend(sorted(list(set([i.get('candidate_name') for i in st.session_state.user_interviews if i.get('candidate_name')])))
        )

        feedback_candidate = st.selectbox("Select Candidate for Feedback", candidate_options, key="feedback_cand_select")
        if feedback_candidate == "New Candidate...":
            feedback_candidate_name = st.text_input("Enter Candidate Name", key="new_feedback_cand_name")
        else:
            feedback_candidate_name = feedback_candidate

        feedback_interviewer = st.text_input("Your Name (Interviewer)", value=st.session_state.get('username', 'Anonymous'), key="feedback_interviewer_name")
        feedback_rating = st.slider("Overall Rating (1-5, 5=Strong Hire)", 1, 5, 3, key="feedback_rating")
        feedback_comments = st.text_area("Comments", height=100, key="feedback_comments")

        if st.button("Submit Feedback", key="submit_feedback_button"):
            if feedback_candidate_name and feedback_interviewer and feedback_comments:
                feedback_data = {
                    "candidate_name": feedback_candidate_name,
                    "interviewer_name": feedback_interviewer,
                    "rating": feedback_rating,
                    "comments": feedback_comments,
                    "timestamp": datetime.now()
                }
                success, response = add_document_to_firestore_collection(
                    f"artifacts/{app_id}/users/{user_email}/interview_feedback", feedback_data, FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL
                )
                if success:
                    st.success(f"Feedback submitted for {feedback_candidate_name} by {feedback_interviewer} with rating {feedback_rating}. (Data saved to Firebase)")
                    log_user_action(user_email, "INTERVIEW_FEEDBACK_SUBMITTED_FIREBASE", {"candidate": feedback_candidate_name, "rating": feedback_rating})
                    st.rerun() # Rerun to refresh feedback trends
                else:
                    st.error(f"❌ Failed to save feedback to Firebase: {response}")
                    log_user_action(user_email, "INTERVIEW_FEEDBACK_FAILED_FIREBASE", {"candidate": feedback_candidate_name, "error": response})
            else:
                st.warning("Please fill in all feedback fields.")

        st.markdown("---")
        st.subheader("Overall Interview Feedback Trends")
        if st.session_state.user_feedback:
            feedback_ratings = [f.get('rating') for f in st.session_state.user_feedback if f.get('rating') is not None]
            if feedback_ratings:
                feedback_trend_data = pd.DataFrame({'Rating': feedback_ratings})
                feedback_counts = feedback_trend_data['Rating'].value_counts().sort_index().reset_index()
                feedback_counts.columns = ['Rating', 'Count']

                fig_feedback_trend = px.bar(
                    feedback_counts,
                    x='Rating',
                    y='Count',
                    title='Distribution of Interview Ratings',
                    labels={'Count': 'Number of Ratings', 'Rating': 'Rating (1-5)'},
                    color='Count',
                    color_continuous_scale=px.colors.sequential.Plasma if dark_mode else px.colors.sequential.Viridis
                )
                st.plotly_chart(fig_feedback_trend, use_container_width=True)
                st.caption("This chart shows the aggregated distribution of interview ratings from Firebase.")
            else:
                st.info("No feedback ratings available to display trends.")
        else:
            st.info("No interview feedback data available in Firebase yet.")


    st.markdown("</div>", unsafe_allow_html=True)
