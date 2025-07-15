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

# --- Mock Salary Data (More Realistic and Granular) ---
# This data simulates real-world salary ranges based on role, experience, and location.
# In a real application, this would come from a secure database or external API.
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
                # New: Specific skills match (mock)
                skills_match_score = st.slider("Specific Skills Match (0-100%)", 0, 100, 70, key="pred_skills_match")
            with col_p2:
                education_level = st.selectbox("Highest Education Level", ["High School", "Associate's", "Bachelor's", "Master's", "PhD"], key="pred_edu")
                interview_feedback = st.slider("Interview Feedback (1-5, 5=Excellent)", 1, 5, 3, key="pred_feedback")
                # New: Past company tier (mock)
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

                likelihood = "Low"
                if likelihood_score >= 9:
                    likelihood = "High"
                elif likelihood_score >= 6:
                    likelihood = "Moderate"
                
                st.success(f"**Prediction:** The candidate has a **{likelihood}** likelihood of success in this role.")
                st.write(f"*(Simulated Score: {likelihood_score}/13)*")
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
                    # New: Suggest learning resources
                    st.markdown("##### Suggested Learning Resources for Missing Skills:")
                    for skill in missing_skills:
                        st.write(f"- For **{skill.title()}**: [Coursera Course](https://www.coursera.org/courses?query={skill.replace(' ', '%20')}), [Udemy Course](https://www.udemy.com/courses/search/?src=ukw&q={skill.replace(' ', '%20')}) (Mock Links)")
                else:
                    st.info("Candidate possesses all required skills!")

                if additional_skills:
                    st.info(f"💡 **Additional Skills (not required but present):** {', '.join(additional_skills).title()}")

                log_user_action(user_email, "SKILL_GAP_ANALYSIS_USED", {"required": required_skills_input, "candidate": candidate_skills_input})

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

                    base_min = benchmark_row['min_salary']
                    base_max = benchmark_row['max_salary']
                    avg_bonus_pct = benchmark_row['avg_bonus_pct']
                    avg_equity_pct = benchmark_row['avg_equity_pct']

                    # Calculate total compensation (simulated)
                    # For simplicity, using the average of min/max base for bonus/equity calculation
                    avg_base_salary = (base_min + base_max) / 2
                    estimated_bonus = avg_base_salary * (avg_bonus_pct / 100)
                    estimated_equity = avg_base_salary * (avg_equity_pct / 100) # Assuming equity as annual value

                    total_comp_min = base_min + (base_min * avg_bonus_pct / 100) + (base_min * avg_equity_pct / 100)
                    total_comp_max = base_max + (base_max * avg_bonus_pct / 100) + (base_max * avg_equity_pct / 100)

                    st.markdown("#### Benchmark Results:")
                    st.success(f"**Estimated Compensation Range for '{selected_seniority} {selected_role}' in '{selected_location}' ({years_exp_comp} yrs exp):**")
                    st.write(f"**Base Salary: ₹{base_min:,.0f} - ₹{base_max:,.0f} per annum**")
                    st.write(f"**Total Compensation (incl. Bonus/Equity): ₹{total_comp_min:,.0f} - ₹{total_comp_max:,.0f} per annum (approx.)**")

                    # Visualization of the range
                    salary_range_df = pd.DataFrame({
                        'Component': ['Base Min', 'Base Max', 'Total Comp Min', 'Total Comp Max'],
                        'Salary': [base_min, base_max, total_comp_min, total_comp_max]
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

        st.markdown("---")
        st.write("More DEI metrics (e.g., ethnicity, disability status) and bias detection features could be added here.")
        
        # New: Mock Bias Detection for Job Description
        st.subheader("Job Description Bias Detection (Mock)")
        st.info("Analyze your job description for potentially biased language. (Mock functionality)")
        jd_text_for_bias = st.text_area("Paste Job Description Text for Bias Check:", "We are seeking a highly motivated and aggressive individual to lead our sales team. Must be a rockstar with a proven track record.", height=150, key="jd_bias_check")
        
        if st.button("Check for Bias", key="check_bias_button"):
            biased_terms = []
            if "aggressive" in jd_text_for_bias.lower():
                biased_terms.append("aggressive (gender-coded)")
            if "rockstar" in jd_text_for_bias.lower():
                biased_terms.append("rockstar (can imply age/culture bias)")
            if "ninja" in jd_text_for_bias.lower():
                biased_terms.append("ninja (can imply culture bias)")
            if "guru" in jd_text_for_bias.lower():
                biased_terms.append("guru (can imply age bias)")

            if biased_terms:
                st.warning("⚠️ Potential biased language detected:")
                for term in biased_terms:
                    st.write(f"- `{term}`: Consider using more neutral alternatives.")
                st.markdown("Suggested alternatives: 'driven', 'high-achieving', 'expert', 'specialist'.")
            else:
                st.success("✅ No obvious biased language detected in this text. Great job!")
            log_user_action(user_email, "JD_BIAS_CHECK_USED")

    st.markdown("</div>", unsafe_allow_html=True)
