import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import collections

def custom_reports_page():
    """
    Renders the Custom Reports page with dummy data and interactive report generation.
    """
    st.markdown('<div class="dashboard-header">📈 Custom Reports</div>', unsafe_allow_html=True)
    st.write("This section allows you to generate custom reports based on your AI screening data.")
    st.info("💡 **Tip:** This page is currently using dummy data for demonstration. Connect your actual screening results for real-time insights!")

    # Determine if dark mode is active to adjust plot colors
    dark_mode = st.session_state.get("dark_mode_main", False)

    # --- Dummy Data Generation ---
    @st.cache_data
    def generate_dummy_screening_data(num_records=100):
        """Generates dummy screening data for reports."""
        data = {
            "Candidate Name": [f"Candidate {i+1}" for i in range(num_records)],
            "Score (%)": [int(min(100, max(40, i * 0.8 + 40 + st.session_state.rng.normal(0, 10)))) for i in range(num_records)],
            "Years Experience": [int(min(20, max(0, i * 0.15 + st.session_state.rng.normal(0, 2)))) for i in range(num_records)],
            "JD Used": st.session_state.rng.choice(["Software Engineer", "Data Scientist", "Product Manager", "UX Designer"], num_records),
            "Location": st.session_state.rng.choice(["Bengaluru", "Mumbai", "Delhi", "Chennai", "Hyderabad"], num_records),
            "CGPA (4.0 Scale)": [round(min(4.0, max(2.0, st.session_state.rng.uniform(2.5, 3.8))), 2) if st.session_state.rng.random() > 0.1 else None for _ in range(num_records)],
            "Matched Keywords": [", ".join(st.session_state.rng.choice(["Python", "SQL", "Machine Learning", "Cloud", "Agile", "Leadership", "Communication"], st.session_state.rng.randint(2, 5))) for _ in range(num_records)],
            "Missing Skills": [", ".join(st.session_state.rng.choice(["Project Management", "TensorFlow", "Kubernetes", "Public Speaking"], st.session_state.rng.randint(0, 3))) for _ in range(num_records)],
            "Date Screened": [datetime.date(2025, st.session_state.rng.randint(1, 7), st.session_state.rng.randint(1, 28)) for _ in range(num_records)]
        }
        df = pd.DataFrame(data)
        
        # Add a 'Shortlisted' column based on a dummy threshold
        dummy_shortlist_threshold = 70
        df['Shortlisted'] = df['Score (%)'].apply(lambda x: "Yes" if x >= dummy_shortlist_threshold else "No")
        
        # Add a 'Tag' column for candidate quality
        df['Tag'] = df.apply(lambda row:
            "👑 Exceptional Match" if row['Score (%)'] >= 90 and row['Years Experience'] >= 5 else (
            "🔥 Strong Candidate" if row['Score (%)'] >= 80 and row['Years Experience'] >= 3 else (
            "✨ Promising Fit" if row['Score (%)'] >= 60 else
            "❌ Limited Match")), axis=1)

        return df

    # Initialize a random number generator in session state for consistent dummy data across reruns
    if 'rng' not in st.session_state:
        st.session_state.rng = pd.np.random.default_rng(42) # Seed for reproducibility

    df_dummy_data = generate_dummy_screening_data()

    st.markdown("### 📊 Report Generator")
    report_type = st.selectbox(
        "Select Report Type",
        ["Select a report...", "Candidate Summary", "JD Performance", "Skill Gap Analysis", "Trend Analysis"],
        key="report_type_select"
    )

    if report_type == "Select a report...":
        st.info("Choose a report type from the dropdown above to generate insights.")
    elif report_type == "Candidate Summary":
        st.subheader("📋 Candidate Summary Report")
        st.write("Overview of candidate performance and characteristics.")

        # Filters for Candidate Summary
        col_filters_cs = st.columns(3)
        with col_filters_cs[0]:
            min_score_cs = st.slider("Minimum Score (%)", 0, 100, 70, key="min_score_cs")
        with col_filters_cs[1]:
            min_exp_cs = st.slider("Minimum Experience (Years)", 0, 20, 2, key="min_exp_cs")
        with col_filters_cs[2]:
            selected_jd_cs = st.multiselect("Filter by JD", df_dummy_data['JD Used'].unique(), key="jd_filter_cs")

        filtered_cs_df = df_dummy_data[
            (df_dummy_data['Score (%)'] >= min_score_cs) &
            (df_dummy_data['Years Experience'] >= min_exp_cs)
        ]
        if selected_jd_cs:
            filtered_cs_df = filtered_cs_df[filtered_cs_df['JD Used'].isin(selected_jd_cs)]

        if not filtered_cs_df.empty:
            st.dataframe(filtered_cs_df[['Candidate Name', 'Score (%)', 'Years Experience', 'JD Used', 'Shortlisted', 'Tag', 'Location', 'CGPA (4.0 Scale)']].sort_values(by="Score (%)", ascending=False), use_container_width=True)

            # Visualizations for Candidate Summary
            st.markdown("#### Visualizations:")
            col_cs_viz1, col_cs_viz2 = st.columns(2)
            with col_cs_viz1:
                st.write("##### Score Distribution")
                fig_score_dist = px.histogram(filtered_cs_df, x="Score (%)", nbins=15, title="Candidate Score Distribution",
                                              color_discrete_sequence=[px.colors.qualitative.Plotly[0]] if not dark_mode else [px.colors.qualitative.Dark2[0]])
                st.plotly_chart(fig_score_dist, use_container_width=True)
            with col_cs_viz2:
                st.write("##### Experience Distribution")
                fig_exp_dist = px.histogram(filtered_cs_df, x="Years Experience", nbins=10, title="Years of Experience Distribution",
                                            color_discrete_sequence=[px.colors.qualitative.Plotly[1]] if not dark_mode else [px.colors.qualitative.Dark2[1]])
                st.plotly_chart(fig_exp_dist, use_container_width=True)
        else:
            st.warning("No data found matching the selected filters for Candidate Summary.")

    elif report_type == "JD Performance":
        st.subheader("📊 Job Description Performance Report")
        st.write("Analyze the average score and number of shortlisted candidates per Job Description.")

        jd_performance_df = df_dummy_data.groupby('JD Used').agg(
            Avg_Score=('Score (%)', 'mean'),
            Total_Candidates=('Candidate Name', 'count'),
            Shortlisted_Count=('Shortlisted', lambda x: (x == 'Yes').sum())
        ).reset_index().sort_values(by="Avg_Score", ascending=False)
        jd_performance_df['Avg_Score'] = jd_performance_df['Avg_Score'].round(2)

        if not jd_performance_df.empty:
            st.dataframe(jd_performance_df, use_container_width=True)

            st.markdown("#### Visualizations:")
            col_jd_viz1, col_jd_viz2 = st.columns(2)
            with col_jd_viz1:
                st.write("##### Average Score by JD")
                fig_jd_score = px.bar(jd_performance_df, x='JD Used', y='Avg_Score', title='Average Score per Job Description',
                                      color='Avg_Score', color_continuous_scale=px.colors.sequential.Teal if not dark_mode else px.colors.sequential.Plasma)
                st.plotly_chart(fig_jd_score, use_container_width=True)
            with col_jd_viz2:
                st.write("##### Shortlisted Count by JD")
                fig_jd_shortlist = px.bar(jd_performance_df, x='JD Used', y='Shortlisted_Count', title='Shortlisted Candidates per Job Description',
                                          color='Shortlisted_Count', color_continuous_scale=px.colors.sequential.Viridis if not dark_mode else px.colors.sequential.Cividis)
                st.plotly_chart(fig_jd_shortlist, use_container_width=True)
        else:
            st.warning("No data available for JD Performance Report.")

    elif report_type == "Skill Gap Analysis":
        st.subheader("🧠 Skill Gap Analysis Report")
        st.write("Identify the most common skills missing from your candidate pool based on screenings.")

        all_missing_skills = []
        for skills_str in df_dummy_data['Missing Skills'].dropna():
            all_missing_skills.extend([s.strip() for s in skills_str.split(',') if s.strip()])
        
        if all_missing_skills:
            missing_skill_counts = collections.Counter(all_missing_skills)
            missing_skills_df = pd.DataFrame(missing_skill_counts.items(), columns=['Missing Skill', 'Frequency']).sort_values(by="Frequency", ascending=False)
            
            st.dataframe(missing_skills_df, use_container_width=True)

            st.markdown("#### Visualizations:")
            fig_missing_skills = px.bar(missing_skills_df.head(10), x='Frequency', y='Missing Skill', orientation='h',
                                        title='Top 10 Most Frequent Missing Skills',
                                        color='Frequency', color_continuous_scale=px.colors.sequential.Sunset if not dark_mode else px.colors.sequential.Inferno)
            st.plotly_chart(fig_missing_skills, use_container_width=True)
        else:
            st.info("No 'Missing Skills' data to analyze for Skill Gap Report. All candidates seem to have required skills!")

    elif report_type == "Trend Analysis":
        st.subheader("📈 Screening Trend Analysis")
        st.write("Analyze screening activity and average scores over time.")

        # Ensure Date Screened is datetime object
        df_dummy_data['Date Screened'] = pd.to_datetime(df_dummy_data['Date Screened'])
        
        # Group by week or month
        time_granularity = st.radio("Group by:", ["Daily", "Weekly", "Monthly"], horizontal=True, key="trend_granularity")
        
        if time_granularity == "Daily":
            df_trends = df_dummy_data.groupby('Date Screened').agg(
                Avg_Score=('Score (%)', 'mean'),
                Total_Screenings=('Candidate Name', 'count')
            ).reset_index()
        elif time_granularity == "Weekly":
            df_trends = df_dummy_data.set_index('Date Screened').resample('W').agg(
                Avg_Score=('Score (%)', 'mean'),
                Total_Screenings=('Candidate Name', 'count')
            ).reset_index()
            df_trends['Date Screened'] = df_trends['Date Screened'].dt.to_period('W').astype(str) # For better display
        else: # Monthly
            df_trends = df_dummy_data.set_index('Date Screened').resample('M').agg(
                Avg_Score=('Score (%)', 'mean'),
                Total_Screenings=('Candidate Name', 'count')
            ).reset_index()
            df_trends['Date Screened'] = df_trends['Date Screened'].dt.to_period('M').astype(str) # For better display

        if not df_trends.empty:
            st.dataframe(df_trends, use_container_width=True)

            st.markdown("#### Visualizations:")
            col_trend_viz1, col_trend_viz2 = st.columns(2)
            with col_trend_viz1:
                st.write("##### Average Score Over Time")
                fig_avg_score_trend = px.line(df_trends, x='Date Screened', y='Avg_Score', title='Average Screening Score Trend',
                                              markers=True, color_discrete_sequence=[px.colors.qualitative.Plotly[2]] if not dark_mode else [px.colors.qualitative.Dark2[2]])
                st.plotly_chart(fig_avg_score_trend, use_container_width=True)
            with col_trend_viz2:
                st.write("##### Total Screenings Over Time")
                fig_total_screen_trend = px.line(df_trends, x='Date Screened', y='Total_Screenings', title='Total Screenings Trend',
                                                 markers=True, color_discrete_sequence=[px.colors.qualitative.Plotly[3]] if not dark_mode else [px.colors.qualitative.Dark2[3]])
                st.plotly_chart(fig_total_screen_trend, use_container_width=True)
        else:
            st.warning("No screening activity data available for Trend Analysis.")

