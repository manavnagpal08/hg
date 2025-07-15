import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import collections
import numpy as np # Import numpy

def custom_reports_page():
    """
    Renders the Custom Reports page, now connected to actual screening data
    from st.session_state['comprehensive_df'].
    """
    st.markdown('<div class="dashboard-header">📈 Custom Reports</div>', unsafe_allow_html=True)
    st.write("This section allows you to generate custom reports based on your AI screening data.")

    # Determine if dark mode is active to adjust plot colors
    dark_mode = st.session_state.get("dark_mode_main", False)

    # --- Load Actual Screening Data ---
    if 'comprehensive_df' in st.session_state and not st.session_state['comprehensive_df'].empty:
        df_screening_data = st.session_state['comprehensive_df'].copy()
        st.success("✅ Loaded actual screening results for reporting.")
    else:
        st.warning("⚠️ No actual screening data found in this session. Please run the Resume Screener first to generate data for reports.")
        st.info("You can still explore the report types, but they will show empty data or require data to be present.")
        df_screening_data = pd.DataFrame() # Ensure an empty DataFrame to prevent errors

    if df_screening_data.empty:
        st.info("No data available for reports. Please screen some resumes or upload data.")
        # Optionally, provide a way to load data or go to screener
        if st.button("Go to Resume Screener"):
            st.session_state.tab_override = '🧠 Resume Screener'
            st.rerun()
        return # Stop execution if no data

    # Ensure 'Date Screened' column is datetime for trend analysis
    if 'Date Screened' in df_screening_data.columns:
        df_screening_data['Date Screened'] = pd.to_datetime(df_screening_data['Date Screened'], errors='coerce')
        df_screening_data.dropna(subset=['Date Screened'], inplace=True) # Remove rows where date conversion failed
    else:
        st.warning("The 'Date Screened' column is missing from your screening data. Trend Analysis will not be available.")
        # Add a dummy date column if missing, for demonstration purposes if needed
        # For actual data, ensure your screener populates this.
        df_screening_data['Date Screened'] = datetime.date.today() # Fallback to today's date


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
            min_exp_cs = st.slider("Minimum Experience (Years)", 0, int(df_screening_data['Years Experience'].max()), 2, key="min_exp_cs")
        with col_filters_cs[2]:
            # Ensure 'JD Used' exists before trying to get unique values
            if 'JD Used' in df_screening_data.columns and not df_screening_data['JD Used'].empty:
                selected_jd_cs = st.multiselect("Filter by JD", df_screening_data['JD Used'].unique(), key="jd_filter_cs")
            else:
                selected_jd_cs = []
                st.info("No 'JD Used' data available for filtering.")


        filtered_cs_df = df_screening_data[
            (df_screening_data['Score (%)'] >= min_score_cs) &
            (df_screening_data['Years Experience'] >= min_exp_cs)
        ]
        if selected_jd_cs:
            filtered_cs_df = filtered_cs_df[filtered_cs_df['JD Used'].isin(selected_jd_cs)]

        if not filtered_cs_df.empty:
            display_cols_cs = ['Candidate Name', 'Score (%)', 'Years Experience', 'JD Used', 'Shortlisted', 'Tag', 'Location', 'CGPA (4.0 Scale)']
            # Filter to only include columns that actually exist in the DataFrame
            display_cols_cs = [col for col in display_cols_cs if col in filtered_cs_df.columns]
            st.dataframe(filtered_cs_df[display_cols_cs].sort_values(by="Score (%)", ascending=False), use_container_width=True)

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

        if 'JD Used' in df_screening_data.columns and not df_screening_data['JD Used'].empty:
            jd_performance_df = df_screening_data.groupby('JD Used').agg(
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
        else:
            st.info("No 'JD Used' data available in screening results for this report.")


    elif report_type == "Skill Gap Analysis":
        st.subheader("🧠 Skill Gap Analysis Report")
        st.write("Identify the most common skills missing from your candidate pool based on screenings.")

        if 'Missing Skills' in df_screening_data.columns and not df_screening_data['Missing Skills'].empty:
            all_missing_skills = []
            for skills_str in df_screening_data['Missing Skills'].dropna():
                all_missing_skills.extend([s.strip() for s in str(skills_str).split(',') if s.strip()]) # Ensure str conversion
            
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
        else:
            st.info("No 'Missing Skills' column found in screening results for this report.")

    elif report_type == "Trend Analysis":
        st.subheader("📈 Screening Trend Analysis")
        st.write("Analyze screening activity and average scores over time.")

        if 'Date Screened' in df_screening_data.columns and not df_screening_data['Date Screened'].empty:
            # Group by week or month
            time_granularity = st.radio("Group by:", ["Daily", "Weekly", "Monthly"], horizontal=True, key="trend_granularity")
            
            # Ensure 'Date Screened' is truly datetime before resampling
            df_screening_data['Date Screened'] = pd.to_datetime(df_screening_data['Date Screened'], errors='coerce')
            df_trends = df_screening_data.dropna(subset=['Date Screened']) # Drop rows where date conversion failed

            if time_granularity == "Daily":
                df_trends = df_trends.groupby('Date Screened').agg(
                    Avg_Score=('Score (%)', 'mean'),
                    Total_Screenings=('Candidate Name', 'count')
                ).reset_index()
            elif time_granularity == "Weekly":
                df_trends = df_trends.set_index('Date Screened').resample('W').agg(
                    Avg_Score=('Score (%)', 'mean'),
                    Total_Screenings=('Candidate Name', 'count')
                ).reset_index()
                df_trends['Date Screened'] = df_trends['Date Screened'].dt.to_period('W').astype(str) # For better display
            else: # Monthly
                df_trends = df_trends.set_index('Date Screened').resample('M').agg(
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
                st.warning("No sufficient screening activity data available for Trend Analysis after date processing.")
        else:
            st.info("The 'Date Screened' column is required for Trend Analysis and was not found in your data.")

