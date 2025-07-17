import streamlit as st
import pandas as pd
import zipfile
import os
import tempfile
import shutil
from datetime import datetime, date
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import collections
import urllib.parse
import uuid

# Import all necessary functions and constants from screener.py
# This ensures consistency in resume parsing and scoring logic
from screener import (
    extract_text_from_file, extract_years_of_experience, extract_email,
    extract_phone_number, extract_location, extract_name, extract_cgpa,
    extract_education_text, extract_work_history, extract_project_details,
    extract_languages, format_work_history, format_project_details,
    generate_concise_ai_suggestion, generate_detailed_hr_assessment,
    semantic_score, MASTER_SKILLS, SKILL_CATEGORIES, create_mailto_link,
    generate_certificate_pdf, send_certificate_email, generate_certificate_html,
    get_tesseract_cmd # Important for OCR setup
)

# Ensure Tesseract is configured for OCR
tesseract_cmd_path = get_tesseract_cmd()
if not tesseract_cmd_path:
    st.error("Tesseract OCR engine not found. Please ensure it's installed and in your system's PATH.")
    st.info("On Streamlit Community Cloud, ensure you have a `packages.txt` file in your repository's root with `tesseract-ocr` and `tesseract-ocr-eng` listed.")
    # st.stop() # Do not stop the entire app, just warn for this page

def bulk_upload_page(comprehensive_df: pd.DataFrame, jd_texts: dict):
    st.title("📦 Bulk Resume Import & Screening")
    st.markdown("Upload a ZIP file containing multiple resumes (PDF, JPG, PNG) for automated batch processing.")

    if 'bulk_comprehensive_df' not in st.session_state:
        st.session_state['bulk_comprehensive_df'] = pd.DataFrame(columns=[
            "File Name", "Candidate Name", "Score (%)", "Years Experience", "CGPA (4.0 Scale)",
            "Email", "Phone Number", "Location", "Languages", "Education Details",
            "Work History", "Project Details", "AI Suggestion", "Detailed HR Assessment",
            "Matched Keywords", "Missing Skills", "Matched Keywords (Categorized)",
            "Missing Skills (Categorized)", "Semantic Similarity", "Resume Raw Text",
            "JD Used", "Date Screened", "Certificate ID", "Certificate Rank", "Tag"
        ])

    st.markdown("## ⚙️ Define Job Requirements & Screening Criteria for Bulk Upload")
    col1, col2 = st.columns([2, 1])

    with col1:
        jd_text = ""
        job_roles = {"Upload my own": None}
        # Populate job_roles from the jd_texts dictionary passed from main.py
        for jd_name, jd_content in jd_texts.items():
            job_roles[jd_name] = jd_content # Store content directly for easier access

        jd_option = st.selectbox("📌 **Select a Pre-Loaded Job Role or Upload Your Own Job Description**", list(job_roles.keys()), key="bulk_jd_select")
        
        jd_name_for_results = ""
        if jd_option == "Upload my own":
            jd_file = st.file_uploader("Upload Job Description (TXT, PDF)", type=["txt", "pdf"], help="Upload a .txt or .pdf file containing the job description.", key="bulk_jd_file_uploader")
            if jd_file:
                jd_text = extract_text_from_file(jd_file)
                jd_name_for_results = jd_file.name.replace('.pdf', '').replace('.txt', '')
            else:
                jd_name_for_results = "Uploaded JD (No file selected)"
        else:
            jd_text = job_roles[jd_option] # Get content directly
            jd_name_for_results = jd_option

        if jd_text:
            with st.expander("📝 View Loaded Job Description"):
                st.text_area("Job Description Content", jd_text, height=200, disabled=True, label_visibility="collapsed")
        else:
            st.warning("Please select or upload a Job Description to proceed with bulk screening.")
            return # Stop execution if no JD is provided

    with col2:
        cutoff = st.slider("📈 **Minimum Score Cutoff (%)**", 0, 100, 75, key="bulk_min_score_cutoff_slider", help="Candidates scoring below this percentage will be flagged for closer review or considered less suitable.")
        min_experience = st.slider("💼 **Minimum Experience Required (Years)**", 0, 15, 2, key="bulk_min_exp_slider", help="Candidates with less than this experience will be noted.")
        max_experience = st.slider("⬆️ **Maximum Experience Allowed (Years)**", 0, 20, 10, key="bulk_max_exp_slider", help="Candidates with more than this experience might be considered overqualified or outside the target range.")
        min_cgpa = st.slider("🎓 **Minimum CGPA Required (4.0 Scale)**", 0.0, 4.0, 2.5, 0.1, key="bulk_min_cgpa_slider", help="Candidates with CGPA below this value (normalized to 4.0) will be noted.")
        st.markdown("---")
        st.info("Once criteria are set, upload your ZIP file below.")

    st.markdown("## 🎯 Skill Prioritization (Optional)")
    st.caption("Assign higher importance to specific skills in the Job Description.")
    
    all_master_skills = sorted(list(MASTER_SKILLS))

    col_weights_1, col_weights_2 = st.columns(2)
    with col_weights_1:
        high_priority_skills = st.multiselect(
            "🌟 **High Priority Skills (Weight x3)**",
            options=all_master_skills,
            help="Select skills that are absolutely critical for this role. These will significantly boost the score if found.",
            key="bulk_high_priority_skills"
        )
    with col_weights_2:
        medium_priority_skills = st.multiselect(
            "✨ **Medium Priority Skills (Weight x2)**",
            options=[s for s in all_master_skills if s not in high_priority_skills],
            help="Select skills that are very important, but not as critical as high priority ones.",
            key="bulk_medium_priority_skills"
        )

    zip_file = st.file_uploader("📂 **Upload Resumes ZIP File**", type=["zip"], help="Upload a .zip file containing multiple PDF, JPG, or PNG resumes.", key="zip_file_uploader")

    if zip_file and jd_text:
        st.markdown("---")
        st.markdown("## 🚀 Processing Resumes...")
        
        # Create a temporary directory to extract files
        temp_dir = tempfile.mkdtemp()
        processed_files = []

        try:
            with zipfile.ZipFile(zip_file, 'r') as zf:
                # Filter for allowed file types within the zip
                allowed_extensions = ['.pdf', '.jpg', '.jpeg', '.png']
                resume_files_in_zip = [
                    f for f in zf.namelist()
                    if not f.startswith('__MACOSX/') and # Exclude Mac specific hidden folders
                       not f.endswith('/') and # Exclude directories
                       os.path.splitext(f.lower())[1] in allowed_extensions
                ]

                if not resume_files_in_zip:
                    st.warning("No valid resume files (PDF, JPG, PNG) found in the uploaded ZIP file.")
                    return

                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []

                for i, file_name_in_zip in enumerate(resume_files_in_zip):
                    status_text.text(f"Extracting and processing: {file_name_in_zip} ({i+1}/{len(resume_files_in_zip)})...")
                    
                    # Extract the file to the temporary directory
                    extracted_path = zf.extract(file_name_in_zip, temp_dir)
                    
                    # Read the extracted file content as bytes for extract_text_from_file
                    with open(extracted_path, 'rb') as f_extracted:
                        file_bytes = f_extracted.read()
                    
                    # Create a BytesIO object to simulate an uploaded file for extract_text_from_file
                    from io import BytesIO
                    uploaded_file_mock = BytesIO(file_bytes)
                    uploaded_file_mock.name = os.path.basename(file_name_in_zip)
                    uploaded_file_mock.type = f"application/{os.path.splitext(file_name_in_zip)[1].lstrip('.')}" if os.path.splitext(file_name_in_zip)[1] else "application/octet-stream"
                    
                    text = extract_text_from_file(uploaded_file_mock)
                    
                    if text.startswith("[ERROR]"):
                        st.error(f"Failed to process {file_name_in_zip}: {text.replace('[ERROR] ', '')}")
                        continue

                    # Reuse existing screening logic
                    exp = extract_years_of_experience(text)
                    email = extract_email(text)
                    phone = extract_phone_number(text)
                    location = extract_location(text)
                    languages = extract_languages(text)
                    education_details_text = extract_education_text(text)
                    work_history_raw = extract_work_history(text)
                    project_details_raw = extract_project_details(text, MASTER_SKILLS)
                    
                    education_details_formatted = education_details_text
                    work_history_formatted = format_work_history(work_history_raw)
                    project_details_formatted = format_project_details(project_details_raw)

                    candidate_name = extract_name(text) or os.path.basename(file_name_in_zip).replace('.pdf', '').replace('.jpg', '').replace('.jpeg', '').replace('.png', '').replace('_', ' ').title()
                    cgpa = extract_cgpa(text)

                    resume_raw_skills_set, resume_categorized_skills = extract_relevant_keywords(text, MASTER_SKILLS)
                    jd_raw_skills_set, jd_categorized_skills = extract_relevant_keywords(jd_text, MASTER_SKILLS)

                    matched_keywords = list(resume_raw_skills_set.intersection(jd_raw_skills_set))
                    missing_skills = list(jd_raw_skills_set.difference(resume_raw_skills_set))

                    score, semantic_similarity = semantic_score(text, jd_text, exp, cgpa, high_priority_skills, medium_priority_skills)
                    
                    concise_ai_suggestion = generate_concise_ai_suggestion(
                        candidate_name=candidate_name,
                        score=score,
                        years_exp=exp,
                        semantic_similarity=semantic_similarity,
                        cgpa=cgpa
                    )

                    detailed_hr_assessment = generate_detailed_hr_assessment(
                        candidate_name=candidate_name,
                        score=score,
                        years_exp=exp,
                        semantic_similarity=semantic_similarity,
                        cgpa=cgpa,
                        jd_text=jd_text,
                        resume_text=text,
                        matched_keywords=matched_keywords,
                        missing_skills=missing_skills,
                        max_exp_cutoff=max_experience
                    )

                    certificate_id = str(uuid.uuid4())
                    certificate_rank = "Not Applicable"

                    if score >= 90:
                        certificate_rank = "🏅 Elite Match"
                    elif score >= 80:
                        certificate_rank = "⭐ Strong Match"
                    elif score >= 75:
                        certificate_rank = "✅ Good Fit"

                    results.append({
                        "File Name": os.path.basename(file_name_in_zip),
                        "Candidate Name": candidate_name,
                        "Score (%)": score,
                        "Years Experience": exp,
                        "CGPA (4.0 Scale)": cgpa,
                        "Email": email or "Not Found",
                        "Phone Number": phone or "Not Found",
                        "Location": location or "Not Found",
                        "Languages": languages,
                        "Education Details": education_details_formatted,
                        "Work History": work_history_formatted,
                        "Project Details": project_details_formatted,
                        "AI Suggestion": concise_ai_suggestion,
                        "Detailed HR Assessment": detailed_hr_assessment,
                        "Matched Keywords": ", ".join(matched_keywords),
                        "Missing Skills": ", ".join(missing_skills),
                        "Matched Keywords (Categorized)": dict(resume_categorized_skills),
                        "Missing Skills (Categorized)": dict(jd_categorized_skills),
                        "Semantic Similarity": semantic_similarity,
                        "Resume Raw Text": text,
                        "JD Used": jd_name_for_results,
                        "Date Screened": datetime.now().date(),
                        "Certificate ID": certificate_id,
                        "Certificate Rank": certificate_rank
                    })
                    progress_bar.progress((i + 1) / len(resume_files_in_zip))
                
                st.session_state['bulk_comprehensive_df'] = pd.DataFrame(results).sort_values(by="Score (%)", ascending=False).reset_index(drop=True)
                
                st.session_state['bulk_comprehensive_df']['Tag'] = st.session_state['bulk_comprehensive_df'].apply(lambda row: 
                    "👑 Exceptional Match" if row['Score (%)'] >= 90 and row['Years Experience'] >= 5 and row['Years Experience'] <= max_experience and row['Semantic Similarity'] >= 0.85 and (row['CGPA (4.0 Scale)'] is None or row['CGPA (4.0 Scale)'] >= 3.5) else (
                    "🔥 Strong Candidate" if row['Score (%)'] >= 80 and row['Years Experience'] >= 3 and row['Years Experience'] <= max_experience and row['Semantic Similarity'] >= 0.7 and (row['CGPA (4.0 Scale)'] is None or row['CGPA (4.0 Scale)'] >= 3.0) else (
                    "✨ Promising Fit" if row['Score (%)'] >= 60 and row['Years Experience'] >= 1 and row['Years Experience'] <= max_experience and (row['CGPA (4.0 Scale)'] is None or row['CGPA (4.0 Scale)'] >= 2.5) else (
                    "⚠️ Needs Review" if row['Score (%)'] >= 40 else 
                    "❌ Limited Match"))), axis=1)

                st.success(f"✅ Successfully processed {len(results)} resumes from the ZIP file!")
                progress_bar.empty()
                status_text.empty()

        except zipfile.BadZipFile:
            st.error("❌ The uploaded file is not a valid ZIP file.")
        except Exception as e:
            st.error(f"An unexpected error occurred during ZIP processing: {e}")
        finally:
            # Clean up the temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                st.info("Cleaned up temporary files.")


    st.markdown("---")
    st.markdown("## 📋 Bulk Screening Results")
    st.caption("Review the results of the batch resume screening.")

    if not st.session_state['bulk_comprehensive_df'].empty:
        # Get the latest shortlist_threshold from the slider
        shortlist_threshold = cutoff # Use the cutoff defined on this page

        filtered_display_df = st.session_state['bulk_comprehensive_df'].copy()
        filtered_display_df['Shortlisted'] = filtered_display_df['Score (%)'].apply(lambda x: f"Yes (Score >= {shortlist_threshold}%)" if x >= shortlist_threshold else "No")

        st.dataframe(
            filtered_display_df[[
                'Candidate Name', 'Score (%)', 'Years Experience', 'CGPA (4.0 Scale)',
                'Email', 'Location', 'Tag', 'AI Suggestion', 'Certificate Rank', 'Shortlisted'
            ]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score (%)": st.column_config.ProgressColumn(
                    "Score (%)",
                    help="Matching score against job requirements",
                    format="%.1f",
                    min_value=0,
                    max_value=100,
                ),
                "Years Experience": st.column_config.NumberColumn(
                    "Years Experience",
                    help="Total years of professional experience",
                    format="%.1f years",
                ),
                "CGPA (4.0 Scale)": st.column_config.NumberColumn(
                    "CGPA (4.0 Scale)",
                    help="Candidate's CGPA normalized to a 4.0 scale",
                    format="%.2f",
                    min_value=0.0,
                    max_value=4.0
                ),
                "AI Suggestion": st.column_config.Column(
                    "AI Suggestion",
                    help="AI's concise overall assessment and recommendation"
                ),
                "Certificate Rank": st.column_config.Column(
                    "Certificate Rank",
                    help="ScreenerPro Certification Level",
                    width="small"
                ),
                "Shortlisted": st.column_config.Column(
                    "Shortlisted",
                    help="Indicates if the candidate meets the defined screening criteria"
                )
            }
        )

        @st.cache_data
        def convert_df_to_csv(df_to_convert):
            return df_to_convert.to_csv(index=False).encode('utf-8')

        csv_data = convert_df_to_csv(st.session_state['bulk_comprehensive_df'])
        st.download_button(
            label="⬇️ Download All Bulk Results as CSV",
            data=csv_data,
            file_name="bulk_screening_results.csv",
            mime="text/csv",
            key="download_bulk_csv_button"
        )
    else:
        st.info("No bulk screening results to display yet. Upload a ZIP file to get started!")

