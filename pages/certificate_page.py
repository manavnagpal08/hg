import streamlit as st
import urllib.parse
from datetime import datetime

st.set_page_config(layout="centered", page_title="Your Certificate")

# Get query parameters from the URL
query_params = st.query_params

# Extract data from query parameters
certificate_id = query_params.get("id", "N/A")
candidate_name = urllib.parse.unquote(query_params.get("name", "Candidate"))
score = query_params.get("score", "N/A")
date = query_params.get("date", datetime.now().strftime("%Y-%m-%d"))
jd_used = urllib.parse.unquote(query_params.get("jd_used", "General Job Description"))

st.title("🎖️ Certificate of Assessment")
st.markdown("---")

st.write(f"This certifies that:")
st.markdown(f"## **{candidate_name}**")

st.write(f"Has successfully completed an AI-powered resume assessment for the role:")
st.markdown(f"### **{jd_used}**")

st.write(f"Achieving an overall match score of:")
st.markdown(f"## **{score}%**")

st.markdown(f"---")
st.write(f"**Date of Assessment:** {date}")
st.write(f"**Certificate ID:** `{certificate_id}`")

st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        font-size: 1.2rem;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        background-color: #4CAF50;
        color: white;
        border: none;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1 {
        text-align: center;
        color: #4CAF50;
        font-size: 3em;
    }
    h2, h3, h4 {
        text-align: center;
    }
    .stMarkdown p {
        text-align: center;
        font-size: 1.2em;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("---")
st.write("This certificate is generated automatically based on the assessment results.")

# Add a download button (optional, but good for certificates)
# You might need a separate mechanism to render this as a PDF or image
# For now, it's just a button that could trigger a download in a more complex setup
if st.button("Download Certificate (Coming Soon)", disabled=True):
    st.info("Download functionality will be implemented in a future update!")
