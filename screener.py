import streamlit as st
import pandas as pd
import os
import io
import docx
from PyPDF2 import PdfReader
import collections
import re

# Placeholder for LLM API call (replace with actual implementation)
# For now, this will return mock data.
async def call_llm_for_screening(resume_text, jd_text):
    """
    Mocks an asynchronous call to an LLM for resume screening.
    In a real application, this would involve a fetch call to a Gemini API endpoint.
    """
    await st.spinner("Analyzing resume with AI...") # Simulate AI processing time

    # Mock LLM response for demonstration purposes
    mock_score = 75 # Example score
    mock_experience = 5 # Example experience
    mock_matched_keywords = "Python, SQL, Data Analysis, Machine Learning"
    mock_missing_skills = "Cloud Computing, Big Data, Leadership"
    mock_ai_suggestion = "Good technical skills, suggest leadership training."
    
    # Extracting basic info from resume_text (mocked)
    mock_candidate_name = "John Doe"
    mock_email = "john.doe@example.com"
    mock_phone = "123-456-7890"
    mock_location = "San Francisco, CA"
    mock_languages = "English, Spanish"
    mock_education = "M.Sc. Computer Science, B.Tech. Electrical Engineering"
    mock_work_history = "5 years as Data Scientist at TechCorp"
    mock_project_details = "Developed AI-powered recommendation system"
    mock_cgpa = 3.8

    # Simulate semantic similarity and categorized keywords if needed
    mock_semantic_similarity = 0.75
    mock_categorized_keywords = {
        "Programming Languages": ["Python", "SQL"],
        "Tools & Technologies": ["Pandas", "Scikit-learn"],
        "Concepts": ["Machine Learning", "Statistical Modeling"]
    }

    return {
        "Score (%)": mock_score,
        "Years Experience": mock_experience,
        "Matched Keywords": mock_matched_keywords,
        "Missing Skills": mock_missing_skills,
        "AI Suggestion": mock_ai_suggestion,
        "Candidate Name": mock_candidate_name,
        "Email": mock_email,
        "Phone Number": mock_phone,
        "Location": mock_location,
        "Languages": mock_languages,
        "Education Details": mock_education,
        "Work History": mock_work_history,
        "Project Details": mock_project_details,
        "CGPA (4.0 Scale)": mock_cgpa,
        "Semantic Similarity": mock_semantic_similarity,
        "Matched Keywords (Categorized)": mock_categorized_keywords,
        "JD Used": "Mock JD" # Indicate a JD was used
    }

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Function to extract text from TXT
def extract_text_from_txt(txt_file):
    return txt_file.read().decode("utf-8")

# Function to load JDs from the 'data' directory
def load_jds():
    jd_dir = "data"
    if not os.path.exists(jd_dir):
        os.makedirs(jd_dir)
    jds = {}
    for filename in os.listdir(jd_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(jd_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                jds[filename] = f.read()
    return jds

# Main resume screener page function
def resume_screener_page():
    st.markdown("## 🧠 AI Resume Screener")
    st.write("Upload resumes and select a Job Description to find the best matches.")

    # Create the 'data' directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")

    jds = load_jds()
    jd_options = list(jds.keys())

    if not jd_options:
        st.warning("No Job Descriptions found. Please go to 'Manage JDs' to upload some.")
        return

    selected_jd_file = st.selectbox("Select a Job Description", jd_options, key="selected_jd_screener")
    selected_jd_content = jds.get(selected_jd_file, "")

    st.markdown("### Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload Resume Files (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="resume_uploader"
    )

    if uploaded_files:
        if st.button("Start Screening", key="start_screening_button"):
            if not selected_jd_content:
                st.error("Please select a Job Description before screening.")
                return

            all_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                file_name = uploaded_file.name
                status_text.text(f"Processing {file_name} ({i+1}/{len(uploaded_files)})...")
                progress_bar.progress((i + 1) / len(uploaded_files))

                try:
                    if file_name.endswith(".pdf"):
                        resume_text = extract_text_from_pdf(uploaded_file)
                    elif file_name.endswith(".docx"):
                        resume_text = extract_text_from_docx(uploaded_file)
                    elif file_name.endswith(".txt"):
                        resume_text = extract_text_from_txt(uploaded_file)
                    else:
                        st.warning(f"Unsupported file type for {file_name}. Skipping.")
                        continue

                    # Call LLM for screening (mocked for now)
                    # In a real app, this would be an actual API call
                    screening_result = st.session_state._run_llm_screening(
                        resume_text, selected_jd_content
                    )
                    
                    # Add file name and JD used to the result
                    screening_result["File Name"] = file_name
                    screening_result["JD Used"] = selected_jd_file
                    all_results.append(screening_result)

                except Exception as e:
                    st.error(f"Error processing {file_name}: {e}")
                    # Append an entry with error for tracking
                    all_results.append({
                        "File Name": file_name,
                        "Candidate Name": "Error",
                        "Score (%)": 0,
                        "Years Experience": 0,
                        "Matched Keywords": "N/A",
                        "Missing Skills": "N/A",
                        "AI Suggestion": f"Error: {e}",
                        "Email": "N/A",
                        "Phone Number": "N/A",
                        "Location": "N/A",
                        "Languages": "N/A",
                        "Education Details": "N/A",
                        "Work History": "N/A",
                        "Project Details": "N/A",
                        "CGPA (4.0 Scale)": None,
                        "Semantic Similarity": None,
                        "Matched Keywords (Categorized)": {},
                        "JD Used": selected_jd_file
                    })
            
            status_text.text("Screening complete!")
            progress_bar.empty()

            if all_results:
                # Convert results to DataFrame
                df_results = pd.DataFrame(all_results)
                
                # Store the comprehensive results in session state
                if 'comprehensive_df' not in st.session_state:
                    st.session_state['comprehensive_df'] = pd.DataFrame()
                
                # Append new results to the existing DataFrame
                st.session_state['comprehensive_df'] = pd.concat(
                    [st.session_state['comprehensive_df'], df_results], ignore_index=True
                )
                
                st.success("✅ Resumes screened successfully!")
                st.markdown("### Screening Results")
                st.dataframe(df_results, use_container_width=True)

                st.markdown("---")
                st.markdown("#### Quick Actions on Results")
                col_actions = st.columns(2)
                with col_actions[0]:
                    if st.button("View Full Analytics Dashboard", key="view_analytics_screener"):
                        st.session_state.tab_override = "📊 Screening Analytics"
                        st.rerun()
                with col_actions[1]:
                    if st.button("Email Shortlisted Candidates", key="email_candidates_screener"):
                        st.session_state.tab_override = "📤 Email Candidates"
                        st.rerun()
            else:
                st.warning("No resumes were successfully screened.")
    
    # Define a placeholder function for LLM screening if it doesn't exist
    # This is a temporary measure to prevent errors if the actual LLM integration
    # is not yet complete or is handled elsewhere.
    if '_run_llm_screening' not in st.session_state:
        st.session_state._run_llm_screening = call_llm_for_screening

# Example of how to integrate the LLM call:
# In your main app, before calling resume_screener_page(), you might set up the LLM
# st.session_state._run_llm_screening = your_actual_llm_call_function
