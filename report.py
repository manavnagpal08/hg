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
        if st.button("Go to Resume Screener"):
            st.session_state.tab_override = '🧠 Resume Screener'
            st.rerun()
        return # Stop execution if no data

    # --- Ensure essential columns exist and handle missing ones gracefully ---
    # Define all expected columns and their default values if missing
    expected_cols = {
        'Score (%)': 0.0,
        'Years Experience': 0.0,
        'File Name': 'Unknown File',
        'Candidate Name': 'Unknown Candidate',
        'JD Used': 'Unknown JD',
        'Location': 'Unknown Location',
        'CGPA (4.0 Scale)': np.nan, # Use NaN for numerical missing values
        'Matched Keywords': '',
        'Missing Skills': '',
        'AI Suggestion': '',
        'Email': '',
        'Phone Number': '',
        'Languages': '',
        'Education Details': '',
        'Work History': '',
        'Project Details': '',
        'Semantic Similarity': np.nan,
        'Matched Keywords (Categorized)': {},
        'Date Screened': datetime.date.today() # Default to today if missing
    }

    for col, default_val in expected_cols.items():
        if col not in df_screening_data.columns:
            df_screening_data[col] = default_val
            st.warning(f"Column '{col}' was missing in screening data and has been added with default values. For accurate reports, ensure your screener populates this column.")
        # Ensure correct data types for critical columns if they exist but might be wrong
        if col in ['Score (%)', 'Years Experience', 'CGPA (4.0 Scale)', 'Semantic Similarity']:
            # Convert to numeric, coercing errors will turn invalid parsing into NaN
            df_screening_data[col] = pd.to_numeric(df_screening_data[col], errors='coerce')
        elif col == 'Date Screened':
            # Convert to datetime, then extract date part, coerce errors to NaT (Not a Time)
            df_screening_data[col] = pd.to_datetime(df_screening_data[col], errors='coerce').dt.date
            # Fill NaT values with today's date
            df_screening_data[col].fillna(datetime.date.today(), inplace=True) 

    # Re-calculate 'Shortlisted' and 'Tag' if they are missing or for consistency
    # These thresholds should ideally be configurable or come from the main app's session state
    dummy_shortlist_threshold = st.session_state.get('screening_cutoff_score', 75)
    df_screening_data['Shortlisted'] = df_screening_data['Score (%)'].apply(lambda x: "Yes" if x >= dummy_shortlist_threshold else "No")
    
    # Re-calculate 'Tag' for consistency
    df_screening_data['Tag'] = df_screening_data.apply(lambda row:
        "👑 Exceptional Match" if row['Score (%)'] >= 90 and row['Years Experience'] >= 5 and row.get('Semantic Similarity', 0) >= 0.85 else (
        "🔥 Strong Candidate" if row['Score (%)'] >= 80 and row['Years Experience'] >= 3 and row.get('Semantic Similarity', 0) >= 0.7 else (
        "✨ Promising Fit" if row['Score (%)'] >= 60 and row['Years Experience'] >= 1 else (
        "⚠️ Needs Review" if row['Score (%)'] >= 40 else
        "❌ Limited Match"))), axis=1)


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
            min_score_cs = st.slider("Minimum Score (%)", 0, 100, 70, key="min_score_cs_report")
        with col_filters_cs[1]:
            # Ensure max value for slider is based on actual data, fallback to a sensible default
            max_exp_val = int(df_screening_data['Years Experience'].max()) if not df_screening_data['Years Experience'].empty else 20
            min_exp_cs = st.slider("Minimum Experience (Years)", 0, max_exp_val, 2, key="min_exp_cs_report")
        with col_filters_cs[2]:
            # Ensure 'JD Used' exists before trying to get unique values
            if 'JD Used' in df_screening_data.columns and not df_screening_data['JD Used'].empty:
                selected_jd_cs = st.multiselect("Filter by JD", df_screening_data['JD Used'].unique(), key="jd_filter_cs_report")
            else:
                selected_jd_cs = []
                st.info("No 'JD Used' data available for filtering.")


        col_filters_cs_2 = st.columns(2)
        with col_filters_cs_2[0]:
            if 'Location' in df_screening_data.columns and not df_screening_data['Location'].empty:
                selected_location_cs = st.multiselect("Filter by Location", df_screening_data['Location'].unique(), key="location_filter_cs_report")
            else:
                selected_location_cs = []
                st.info("No 'Location' data available for filtering.")
        with col_filters_cs_2[1]:
            if 'Tag' in df_screening_data.columns and not df_screening_data['Tag'].empty:
                selected_tag_cs = st.multiselect("Filter by Candidate Tag", df_screening_data['Tag'].unique(), key="tag_filter_cs_report")
            else:
                selected_tag_cs = []
                st.info("No 'Tag' data available for filtering.")


        filtered_cs_df = df_screening_data[
            (df_screening_data['Score (%)'] >= min_score_cs) &
            (df_screening_data['Years Experience'] >= min_exp_cs)
        ]
        if selected_jd_cs:
            filtered_cs_df = filtered_cs_df[filtered_cs_df['JD Used'].isin(selected_jd_cs)]
        if selected_location_cs:
            filtered_cs_df = filtered_cs_df[filtered_cs_df['Location'].isin(selected_location_cs)]
        if selected_tag_cs:
            filtered_cs_df = filtered_cs_df[filtered_cs_df['Tag'].isin(selected_tag_cs)]


        if not filtered_cs_df.empty:
            st.markdown("#### Summary of Filtered Candidates:")
            st.markdown(f"- **Total Candidates:** {len(filtered_cs_df)}")
            st.markdown(f"- **Average Score:** {filtered_cs_df['Score (%)'].mean():.2f}%")
            st.markdown(f"- **Average Years of Experience:** {filtered_cs_df['Years Experience'].mean():.1f} years")

            if 'Tag' in filtered_cs_df.columns and not filtered_cs_df['Tag'].empty:
                tag_counts = filtered_cs_df['Tag'].value_counts()
                st.markdown("- **Candidate Tag Distribution:**")
                for tag, count in tag_counts.items():
                    st.markdown(f"  - {tag}: {count} candidate(s)")
            
            if 'Location' in filtered_cs_df.columns and not filtered_cs_df['Location'].empty:
                location_counts = filtered_cs_df['Location'].value_counts()
                if not location_counts.empty:
                    st.markdown("- **Top Locations:**")
                    for loc, count in location_counts.head(3).items():
                        st.markdown(f"  - {loc}: {count} candidate(s)")

            st.markdown("---")
            st.markdown("#### Individual Candidate Summaries:")
            for idx, row in filtered_cs_df.iterrows():
                st.markdown(f"##### {row['Candidate Name']}")
                st.markdown(f"""
                - **Score:** {row['Score (%)']:.1f}% (Tag: {row['Tag']})
                - **Experience:** {row['Years Experience']:.1f} years
                - **JD Applied:** {row['JD Used']}
                - **Contact:** {row['Email']} | {row['Phone Number']}
                - **Location:** {row['Location']}
                - **AI Insight:** {row['AI Suggestion']}
                """)
                st.markdown("---") # Separator for each candidate

            st.markdown("#### Full Filtered Candidates Table:")
            display_cols_cs = ['Candidate Name', 'Score (%)', 'Years Experience', 'JD Used', 'Shortlisted', 'Tag', 'Location', 'CGPA (4.0 Scale)', 'Email', 'Phone Number']
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
            
            # New visualizations
            col_cs_viz3, col_cs_viz4 = st.columns(2)
            with col_cs_viz3:
                if 'Location' in filtered_cs_df.columns and not filtered_cs_df['Location'].empty:
                    st.write("##### Candidates by Location")
                    location_counts = filtered_cs_df['Location'].value_counts().reset_index()
                    location_counts.columns = ['Location', 'Count']
                    fig_location_pie = px.pie(location_counts, values='Count', names='Location', title='Candidates by Location',
                                              color_discrete_sequence=px.colors.qualitative.Pastel if not dark_mode else px.colors.qualitative.Dark2)
                    st.plotly_chart(fig_location_pie, use_container_width=True)
                else:
                    st.info("No 'Location' data for this visualization.")
            with col_cs_viz4:
                if 'Tag' in filtered_cs_df.columns and not filtered_cs_df['Tag'].empty:
                    st.write("##### Candidates by Quality Tag")
                    tag_counts = filtered_cs_df['Tag'].value_counts().reset_index()
                    tag_counts.columns = ['Tag', 'Count']
                    fig_tag_bar = px.bar(tag_counts, x='Tag', y='Count', title='Candidates by Quality Tag',
                                         color='Count', color_continuous_scale=px.colors.sequential.Plasma if dark_mode else px.colors.sequential.Viridis)
                    st.plotly_chart(fig_tag_bar, use_container_width=True)
                else:
                    st.info("No 'Tag' data for this visualization.")

        else:
            st.warning("No data found matching the selected filters for Candidate Summary.")

    elif report_type == "JD Performance":
        st.subheader("📊 Job Description Performance Report")
        st.write("Analyze the average score and number of shortlisted candidates per Job Description.")

        if 'JD Used' in df_screening_data.columns and not df_screening_data['JD Used'].empty:
            
            if 'Location' in df_screening_data.columns and not df_screening_data['Location'].empty:
                selected_location_jd = st.multiselect("Filter by Location", df_screening_data['Location'].unique(), key="location_filter_jd_report")
            else:
                selected_location_jd = []
                st.info("No 'Location' data available for filtering.")
            
            filtered_jd_df = df_screening_data.copy()
            if selected_location_jd:
                filtered_jd_df = filtered_jd_df[filtered_jd_df['Location'].isin(selected_location_jd)]

            if not filtered_jd_df.empty:
                jd_performance_df = filtered_jd_df.groupby('JD Used').agg(
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
                    st.warning("No data available for JD Performance Report after filtering.")
            else:
                st.warning("No data available for JD Performance Report with the selected filters.")
        else:
            st.info("No 'JD Used' data available in screening results for this report.")


    elif report_type == "Skill Gap Analysis":
        st.subheader("🧠 Skill Gap Analysis Report")
        st.write("Identify the most common skills missing from your candidate pool based on screenings.")

        if 'Missing Skills' in df_screening_data.columns and not df_screening_data['Missing Skills'].empty:
            
            if 'JD Used' in df_screening_data.columns and not df_screening_data['JD Used'].empty:
                selected_jd_sga = st.multiselect("Filter by JD", df_screening_data['JD Used'].unique(), key="jd_filter_sga_report")
            else:
                selected_jd_sga = []
                st.info("No 'JD Used' data available for filtering.")
            
            filtered_sga_df = df_screening_data.copy()
            if selected_jd_sga:
                filtered_sga_df = filtered_sga_df[filtered_sga_df['JD Used'].isin(selected_jd_sga)]

            all_missing_skills = []
            for skills_str in filtered_sga_df['Missing Skills'].dropna():
                # Ensure skills_str is a string before splitting
                if isinstance(skills_str, str):
                    all_missing_skills.extend([s.strip() for s in skills_str.split(',') if s.strip()])
                elif isinstance(skills_str, list): # Handle if 'Missing Skills' is already a list
                    all_missing_skills.extend([s.strip() for s in skills_str if s.strip()])
            
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
                st.info("No 'Missing Skills' data to analyze for Skill Gap Report with current filters. All candidates seem to have required skills!")
        else:
            st.info("No 'Missing Skills' column found in screening results for this report.")

    elif report_type == "Trend Analysis":
        st.subheader("📈 Screening Trend Analysis")
        st.write("Analyze screening activity and average scores over time.")

        if 'Date Screened' in df_screening_data.columns and not df_screening_data['Date Screened'].empty and df_screening_data['Date Screened'].nunique() > 1:
            # Group by week or month
            time_granularity = st.radio("Group by:", ["Daily", "Weekly", "Monthly"], horizontal=True, key="trend_granularity_report")
            
            # Ensure 'Date Screened' is truly datetime before resampling
            df_trends = df_screening_data.dropna(subset=['Date Screened']) # Drop rows where date conversion failed
            df_trends['Date Screened'] = pd.to_datetime(df_trends['Date Screened'])

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
                # Format for display, handling cases where the period might be empty
                df_trends['Date Display'] = df_trends['Date Screened'].dt.to_period('W').astype(str)
                df_trends.rename(columns={'Date Screened': 'Original Date'}, inplace=True)
                df_trends.rename(columns={'Date Display': 'Date Screened'}, inplace=True)

            else: # Monthly
                df_trends = df_trends.set_index('Date Screened').resample('M').agg(
                    Avg_Score=('Score (%)', 'mean'),
                    Total_Screenings=('Candidate Name', 'count')
                ).reset_index()
                # Format for display, handling cases where the period might be empty
                df_trends['Date Display'] = df_trends['Date Screened'].dt.to_period('M').astype(str)
                df_trends.rename(columns={'Date Screened': 'Original Date'}, inplace=True)
                df_trends.rename(columns={'Date Display': 'Date Screened'}, inplace=True)


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
            st.info("The 'Date Screened' column is required for Trend Analysis and was not found or has insufficient unique dates in your data.")

