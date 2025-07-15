import streamlit as st
import pandas as pd
import plotly.express as px
import collections
import numpy as np
from datetime import datetime

# --- Logging Function (can be shared or imported from main.py if needed) ---
def log_user_action(user_email, action, details=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if details:
        print(f"LOG: [{timestamp}] User '{user_email}' performed action '{action}' with details: {details}")
    else:
        print(f"LOG: [{timestamp}] User '{user_email}' performed action '{action}'")

# --- Advanced Tools Page Function ---
def advanced_tools_page():
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

    tab_predictive, tab_skill_gap, tab_compensation, tab_dei = st.tabs([
        "🔮 Predictive Analytics", "🧩 Skill Gap Analysis", "💰 Compensation Benchmarking", "🤝 DEI Analytics"
    ])

    with tab_predictive:
        st.subheader("Candidate Success Prediction (Mock)")
        st.info("This tool predicts the likelihood of a candidate succeeding in a role based on various factors. (Mock functionality)")

        with st.form("predictive_form", clear_on_submit=False):
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                candidate_score = st.slider("Candidate Score (%)", 0, 100, 75, key="pred_score")
                years_experience = st.slider("Years of Experience", 0.0, 30.0, 5.0, step=0.5, key="pred_exp")
            with col_p2:
                education_level = st.selectbox("Highest Education Level", ["High School", "Associate's", "Bachelor's", "Master's", "PhD"], key="pred_edu")
                interview_feedback = st.slider("Interview Feedback (1-5, 5=Excellent)", 1, 5, 3, key="pred_feedback")
            
            predict_button = st.form_submit_button("Predict Success")

            if predict_button:
                # Mock prediction logic
                likelihood = "Moderate"
                if candidate_score >= 80 and years_experience >= 5 and interview_feedback >= 4:
                    likelihood = "High"
                elif candidate_score < 60 or years_experience < 2:
                    likelihood = "Low"
                
                st.success(f"**Prediction:** The candidate has a **{likelihood}** likelihood of success in this role.")
                log_user_action(user_email, "PREDICTIVE_ANALYTICS_USED", {"score": candidate_score, "exp": years_experience, "likelihood": likelihood})

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
                else:
                    st.info("Candidate possesses all required skills!")

                if additional_skills:
                    st.info(f"💡 **Additional Skills (not required but present):** {', '.join(additional_skills).title()}")

                log_user_action(user_email, "SKILL_GAP_ANALYSIS_USED", {"required": required_skills_input, "candidate": candidate_skills_input})

    with tab_compensation:
        st.subheader("Compensation Benchmarking (Mock)")
        st.info("Benchmark salary ranges for specific roles based on industry data and location. (Mock functionality)")

        with st.form("compensation_form", clear_on_submit=False):
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                role_title = st.text_input("Role Title", "Software Engineer", key="comp_role")
                location = st.text_input("Location", "Bengaluru, India", key="comp_loc")
            with col_c2:
                years_exp_comp = st.slider("Years of Experience (for benchmark)", 0, 20, 5, key="comp_exp")
                industry = st.selectbox("Industry", ["Tech", "Finance", "Healthcare", "Manufacturing", "Other"], key="comp_industry")
            
            benchmark_button = st.form_submit_button("Get Benchmark")

            if benchmark_button:
                # Mock compensation data
                base_salary = 50000 + (years_exp_comp * 5000)
                if "senior" in role_title.lower():
                    base_salary += 20000
                if "manager" in role_title.lower():
                    base_salary += 30000
                
                if "bengaluru" in location.lower():
                    base_salary *= 1.2 # Higher for tech hubs
                
                min_salary = int(base_salary * 0.9)
                max_salary = int(base_salary * 1.1)

                st.markdown("#### Benchmark Results:")
                st.success(f"**Estimated Salary Range for '{role_title}' in '{location}' ({years_exp_comp} yrs exp):**")
                st.write(f"**₹{min_salary:,.0f} - ₹{max_salary:,.0f} per annum (approx.)**")
                st.caption("_Note: This is a mock benchmark and does not reflect real-time market data._")
                log_user_action(user_email, "COMPENSATION_BENCHMARK_USED", {"role": role_title, "location": location})

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

        st.markdown("---")
        st.write("More DEI metrics (e.g., ethnicity, disability status) and bias detection features could be added here.")
        log_user_action(user_email, "DEI_ANALYTICS_VIEWED")

    st.markdown("</div>", unsafe_allow_html=True)
