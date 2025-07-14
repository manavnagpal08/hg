import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import json
import datetime
import plotly.express as px
import pdfplumber
import re
import numpy as np
import collections
import urllib.parse
import joblib # For loading ML models
from sklearn.metrics.pairwise import cosine_similarity # For semantic similarity
from sentence_transformers import SentenceTransformer # For semantic embeddings
import nltk # For stopwords

# Ensure NLTK stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- Global Configuration and Helpers ---
# Set Streamlit page configuration for wide layout
st.set_page_config(page_title="ScreenerPro – AI Hiring Dashboard", layout="wide", page_icon="🧠")

# --- Helper for Activity Logging ---
def log_activity(message):
    """Logs an activity with a timestamp to the session state."""
    if 'activity_log' not in st.session_state:
        st.session_state.activity_log = []
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.activity_log.insert(0, f"[{timestamp}] {message}") # Add to the beginning for most recent first
    # Keep log size manageable, e.g., last 50 activities
    st.session_state.activity_log = st.session_state.activity_log[:50]

# --- User Management (from login.py) ---
USERS_FILE = "users.json"

def load_users():
    """Loads user data from a JSON file."""
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    """Saves user data to a JSON file."""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def hash_password(password):
    """A simple placeholder for password hashing."""
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_hash, provided_password):
    """Verifies a provided password against a stored hash."""
    return stored_hash == hash_password(provided_password)

def is_current_user_admin():
    """Checks if the currently logged-in user is an admin."""
    if 'username' not in st.session_state:
        return False
    users = load_users()
    user_data = users.get(st.session_state.username)
    return isinstance(user_data, dict) and user_data.get("role") == "admin"

def login_section():
    """Handles user login and authentication."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None

    if st.session_state.authenticated:
        return True

    st.sidebar.subheader("Login")
    username = st.sidebar.text_input("Username (Email)", key="login_username_input")
    password = st.sidebar.text_input("Password", type="password", key="login_password_input")

    if st.sidebar.button("Login", key="login_button"):
        users = load_users()
        user_data = users.get(username)
        if user_data and isinstance(user_data, dict) and user_data.get("status") == "active" and verify_password(user_data["password"], password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.sidebar.success("Logged in successfully!")
            log_activity(f"User '{username}' logged in.")
            st.rerun()
        else:
            st.sidebar.error("Invalid username, password, or account is inactive.")
            log_activity(f"Failed login attempt for '{username}'.")
    
    st.sidebar.markdown("---")
    st.sidebar.info("New users, please register with an administrator.")

    return st.session_state.authenticated

def admin_registration_section():
    """Admin tool to register new users."""
    st.subheader("➕ Register New User")
    with st.form("register_user_form"):
        new_username = st.text_input("New Username (Email)", key="new_user_email")
        new_password = st.text_input("New Password", type="password", key="new_user_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_new_user_password")
        new_role = st.selectbox("Role", ["user", "admin"], key="new_user_role")
        register_button = st.form_submit_button("Register User")

        if register_button:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            elif not new_username or not new_password:
                st.error("Username and password cannot be empty.")
            else:
                users = load_users()
                if new_username in users:
                    st.error("User already exists.")
                else:
                    users[new_username] = {"password": hash_password(new_password), "role": new_role, "status": "active"}
                    save_users(users)
                    st.success(f"User '{new_username}' registered successfully as {new_role}!")
                    log_activity(f"Admin '{st.session_state.username}' registered new user '{new_username}' with role '{new_role}'.")

def admin_password_reset_section():
    """Admin tool to reset user passwords."""
    st.subheader("🔑 Reset User Password")
    with st.form("reset_password_form"):
        target_username = st.text_input("Username to Reset Password For", key="reset_user_email")
        new_password = st.text_input("New Password", type="password", key="reset_new_password")
        confirm_password = st.text_input("Confirm New Password", type="password", key="reset_confirm_password")
        reset_button = st.form_submit_button("Reset Password")

        if reset_button:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            elif not target_username or not new_password:
                st.error("Username and new password cannot be empty.")
            else:
                users = load_users()
                if target_username not in users:
                    st.error("User not found.")
                else:
                    users[target_username]["password"] = hash_password(new_password)
                    save_users(users)
                    st.success(f"Password for '{target_username}' reset successfully!")
                    log_activity(f"Admin '{st.session_state.username}' reset password for user '{target_username}'.")

def admin_disable_enable_user_section():
    """Admin tool to disable/enable user accounts."""
    st.subheader("🚫 Enable/Disable User Account")
    with st.form("toggle_user_status_form"):
        target_username = st.text_input("Username to Enable/Disable", key="toggle_user_email")
        action = st.radio("Action", ["Enable", "Disable"], key="toggle_user_action")
        toggle_button = st.form_submit_button(f"{action} User")

        if toggle_button:
            users = load_users()
            if target_username not in users:
                st.error("User not found.")
            else:
                users[target_username]["status"] = "active" if action == "Enable" else "inactive"
                save_users(users)
                st.success(f"User '{target_username}' account set to '{users[target_username]['status']}'!")
                log_activity(f"Admin '{st.session_state.username}' set status of '{target_username}' to '{users[target_username]['status']}'.")

# --- Feedback & Help (from feedback.py) ---
def feedback_and_help_page():
    st.markdown('<div class="dashboard-header">❓ Feedback & Help</div>', unsafe_allow_html=True)
    st.write("We value your feedback! Please share your thoughts, suggestions, or report any issues you encounter.")

    with st.form("feedback_form"):
        feedback_type = st.radio(
            "Type of Feedback:",
            ["Suggestion", "Bug Report", "General Comment", "Feature Request"],
            key="feedback_type"
        )
        feedback_message = st.text_area("Your Message:", height=150, key="feedback_message")
        contact_email = st.text_input("Your Email (Optional, for follow-up):", value=st.session_state.get('username', ''), key="feedback_email")
        submit_button = st.form_submit_button("Submit Feedback")

        if submit_button:
            if not feedback_message.strip():
                st.error("Please enter your feedback message.")
            else:
                # In a real application, you would save this to a database,
                # send an email, or use an external service.
                # For this example, we'll just log it and show a success message.
                feedback_entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "user": st.session_state.get('username', 'anonymous'),
                    "type": feedback_type,
                    "message": feedback_message,
                    "contact_email": contact_email
                }
                
                # Append to a simple file for demonstration
                try:
                    with open("feedback_log.jsonl", "a") as f:
                        f.write(json.dumps(feedback_entry) + "\n")
                    st.success("Thank you for your feedback! We've received your message.")
                    log_activity(f"User '{st.session_state.get('username', 'anonymous')}' submitted feedback ({feedback_type}).")
                    # Clear the form after submission
                    st.session_state.feedback_message = ""
                    st.session_state.feedback_email = st.session_state.get('username', '') # Reset email to current user's
                except Exception as e:
                    st.error(f"Failed to save feedback: {e}")

    st.markdown("---")
    st.subheader("Help & FAQs")
    st.markdown("""
    Welcome to ScreenerPro! Here are some tips to get started:

    * **Dashboard:** Your central hub for key metrics and quick actions.
    * **Resume Screener:** Upload job descriptions and resumes to get instant matching scores and AI assessments.
    * **Manage JDs:** Add, view, or delete your job description templates.
    * **Screening Analytics:** Dive deeper into the data with interactive charts and detailed candidate breakdowns.
    * **Email Candidates:** Generate pre-filled email templates for shortlisted candidates.
    * **Search Resumes:** Quickly find candidates based on keywords.
    * **Candidate Notes:** Add and manage private notes for each candidate.
    * **Admin Tools:** (Admins only) Manage user accounts, including registration, password resets, and status changes.

    **Troubleshooting Tips:**
    * **"File Not Found" Errors:** Ensure your `ml_screening_model.pkl` and any `.txt` JD files are in the same directory as this script.
    * **PDF Processing Issues:** Some complex PDF formats might not extract text perfectly. Try converting them to a simpler PDF or text format if issues persist.
    * **Performance:** Processing many resumes can take time. Please be patient.
    * **Data Persistence:** Your screening results and notes are stored in the current session. If you close your browser or the app restarts, the data will be cleared unless explicitly saved/loaded (feature not implemented in this version for simplicity).
    """)

# --- ML Model Loading (moved from screener.py) ---
@st.cache_resource
def load_ml_model():
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        # Placeholder for actual ML model. If you have a trained model, uncomment and load it.
        # ml_model = joblib.load("ml_screening_model.pkl") 
        ml_model = None # Set to None for now if no model is provided
        if ml_model is None:
            st.warning("ML screening model ('ml_screening_model.pkl') not found. Semantic scoring will fall back to a basic keyword-based approach. Please train and provide the model for full functionality.")
        return model, ml_model
    except Exception as e:
        st.error(f"❌ Error loading ML models: {e}. Please ensure 'ml_screening_model.pkl' is in the same directory.")
        return None, None

model, ml_model = load_ml_model()

# --- Predefined Lists & Categories (from screener.py) ---
MASTER_CITIES = set([
    # Indian Cities
    "Bengaluru", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "Chandigarh", "Kochi", "Coimbatore", "Nagpur", "Bhopal", "Indore", "Gurgaon", "Noida", "Surat", "Visakhapatnam",
    "Patna", "Vadodara", "Ghaziabad", "Ludhiana", "Agra", "Nashik", "Faridabad", "Meerut", "Rajkot", "Varanasi",
    "Srinagar", "Aurangabad", "Dhanbad", "Amritsar", "Allahabad", "Ranchi", "Jamshedpur", "Gwalior", "Jabalpur",
    "Vijayawada", "Jodhpur", "Raipur", "Kota", "Guwahati", "Thiruvananthapuram", "Mysuru", "Hubballi-Dharwad",
    "Mangaluru", "Belagavi", "Davangere", "Ballari", "Tumakuru", "Shivamogga", "Bidar", "Hassan", "Gadag-Betageri",
    "Chitradurga", "Udupi", "Kolar", "Mandya", "Chikkamagaluru", "Koppal", "Chamarajanagar", "Yadgir", "Raichur",
    "Kalaburagi", "Bengaluru Rural", "Dakshina Kannada", "Uttara Kannada", "Kodagu", "Chikkaballapur", "Ramanagara",
    "Bagalkot", "Gadag", "Haveri", "Vijayanagara", "Krishnagiri", "Vellore", "Salem", "Erode", "Tiruppur", "Madurai",
    "Tiruchirappalli", "Thanjavur", "Dindigul", "Kanyakumari", "Thoothukudi", "Tirunelveli", "Nagercoil", "Puducherry",
    "Panaji", "Margao", "Vasco da Gama", "Mapusa", "Ponda", "Bicholim", "Curchorem", "Sanquelim", "Valpoi", "Pernem",
    "Quepem", "Canacona", "Mormugao", "Sanguem", "Dharbandora", "Tiswadi", "Salcete", "Bardez",

    # Foreign Cities (a selection)
    "London", "New York", "Paris", "Berlin", "Tokyo", "Sydney", "Toronto", "Vancouver", "Singapore", "Dubai",
    "San Francisco", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego",
    "Dallas", "San Jose", "Austin", "Jacksonville", "Fort Worth", "Columbus", "Charlotte", "Indianapolis",
    "Seattle", "Denver", "Washington D.C.", "Boston", "Nashville", "El Paso", "Detroit", "Oklahoma City",
    "Portland", "Las Vegas", "Memphis", "Louisville", "Baltimore", "Milwaukee", "Albuquerque", "Tucson",
    "Fresno", "Sacramento", "Mesa", "Atlanta", "Kansas City", "Colorado Springs", "Raleigh", "Miami", "Omaha",
    "Virginia Beach", "Long Beach", "Oakland", "Minneapolis", "Tulsa", "Wichita", "New Orleans", "Cleveland",
    "Tampa", "Honolulu", "Anaheim", "Santa Ana", "St. Louis", "Riverside", "Lexington", "Pittsburgh", "Cincinnati",
    "Anchorage", "Plano", "Newark", "Orlando", "Irvine", "Garland", "Hialeah", "Scottsdale", "North Las Vegas",
    "Chandler", "Laredo", "Chula Vista", "Madison", "Reno", "Buffalo", "Durham", "Rochester", "Winston-Salem",
    "St. Petersburg", "Jersey City", "Toledo", "Lincoln", "Greensboro", "Boise", "Richmond", "Stockton",
    "San Bernardino", "Des Moines", "Modesto", "Fayetteville", "Shreveport", "Akron", "Tacoma", "Aurora",
    "Oxnard", "Fontana", "Montgomery", "Little Rock", "Grand Rapids", "Springfield", "Yonkers", "Augusta",
    "Mobile", "Port St. Lucie", "Denton", "Spokane", "Chattanooga", "Worcester", "Providence", "Fort Lauderdale",
    "Chesapeake", "Fremont", "Baton Rouge", "Santa Clarita", "Birmingham", "Glendale", "Huntsville",
    "Salt Lake City", "Frisco", "McKinney", "Grand Prairie", "Overland Park", "Brownsville", "Killeen",
    "Pasadena", "Olathe", "Dayton", "Savannah", "Fort Collins", "Naples", "Gainesville", "Lakeland", "Sarasota",
    "Daytona Beach", "Melbourne", "Clearwater", "St. Augustine", "Key West", "Fort Myers", "Cape Coral",
    "Coral Springs", "Pompano Beach", "Miami Beach", "West Palm Beach", "Boca Raton", "Fort Pierce",
    "Port Orange", "Kissimmee", "Sanford", "Ocala", "Bradenton", "Palm Bay", "Deltona", "Largo",
    "Deerfield Beach", "Boynton Beach", "Coconut Creek", "Sunrise", "Plantation", "Davie", "Miramar",
    "Hollywood", "Pembroke Pines", "Coral Gables", "Doral", "Aventura", "Sunny Isles Beach", "North Miami",
    "Miami Gardens", "Homestead", "Cutler Bay", "Pinecrest", "Kendall", "Richmond Heights", "West Kendall",
    "East Kendall", "South Miami", "Sweetwater", "Opa-locka", "Florida City", "Golden Glades", "Leisure City",
    "Princeton", "West Perrine", "Naranja", "Goulds", "South Miami Heights", "Country Walk", "The Crossings",
    "Three Lakes", "Richmond West", "Palmetto Bay", "Palmetto Estates", "Perrine", "Cutler Ridge", "Westview",
    "Gladeview", "Brownsville", "Liberty City", "West Little River", "Pinewood", "Ojus", "Ives Estates",
    "Highland Lakes", "Sunny Isles Beach", "Golden Beach", "Bal Harbour", "Surfside", "Bay Harbor Islands",
    "Indian Creek", "North Bay Village", "Biscayne Park", "El Portal", "Miami Shores", "North Miami Beach",
    "Aventura"
])

NLTK_STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
CUSTOM_STOP_WORDS = set([
    "work", "experience", "years", "year", "months", "month", "day", "days", "project", "projects",
    "team", "teams", "developed", "managed", "led", "created", "implemented", "designed",
    "responsible", "proficient", "knowledge", "ability", "strong", "proven", "demonstrated",
    "solution", "solutions", "system", "systems", "platform", "platforms", "framework", "frameworks",
    "database", "databases", "server", "servers", "cloud", "computing", "machine", "learning",
    "artificial", "intelligence", "api", "apis", "rest", "graphql", "agile", "scrum", "kanban",
    "devops", "ci", "cd", "testing", "qa",
    "security", "network", "networking", "virtualization",
    "containerization", "docker", "kubernetes", "git", "github", "gitlab", "bitbucket", "jira",
    "confluence", "slack", "microsoft", "google", "amazon", "azure", "oracle", "sap", "crm", "erp",
    "salesforce", "servicenow", "tableau", "powerbi", "qlikview", "excel", "word", "powerpoint",
    "outlook", "visio", "html", "css", "js", "web", "data", "science", "analytics", "engineer",
    "software", "developer", "analyst", "business", "management", "reporting", "analysis", "tools",
    "python", "java", "javascript", "c++", "c#", "php", "ruby", "go", "swift", "kotlin", "r",
    "sql", "nosql", "linux", "unix", "windows", "macos", "ios", "android", "mobile", "desktop",
    "application", "applications", "frontend", "backend", "fullstack", "ui", "ux", "design",
    "architecture", "architect", "engineering", "scientist", "specialist", "consultant",
    "associate", "senior", "junior", "lead", "principal", "director", "manager", "head", "chief",
    "officer", "president", "vice", "executive", "ceo", "cto", "cfo", "coo", "hr", "human",
    "resources", "recruitment", "talent", "acquisition", "onboarding", "training", "development",
    "performance", "compensation", "benefits", "payroll", "compliance", "legal", "finance",
    "accounting", "auditing", "tax", "budgeting", "forecasting", "investments", "marketing",
    "sales", "customer", "service", "support", "operations", "supply", "chain", "logistics",
    "procurement", "manufacturing", "production", "quality", "assurance", "control", "research",
    "innovation", "product", "program", "portfolio", "governance", "risk", "communication",
    "presentation", "negotiation", "problem", "solving", "critical", "thinking", "analytical",
    "creativity", "adaptability", "flexibility", "teamwork", "collaboration", "interpersonal",
    "organizational", "time", "multitasking", "detail", "oriented", "independent", "proactive",
    "self", "starter", "results", "driven", "client", "facing", "stakeholder", "engagement",
    "vendor", "budget", "cost", "reduction", "process", "improvement", "standardization",
    "optimization", "automation", "digital", "transformation", "change", "methodologies",
    "industry", "regulations", "regulatory", "documentation", "technical", "writing",
    "dashboards", "visualizations", "workshops", "feedback", "reviews", "appraisals",
    "offboarding", "employee", "relations", "diversity", "inclusion", "equity", "belonging",
    "corporate", "social", "responsibility", "csr", "sustainability", "environmental", "esg",
    "ethics", "integrity", "professionalism", "confidentiality", "discretion", "accuracy",
    "precision", "efficiency", "effectiveness", "scalability", "robustness", "reliability",
    "vulnerability", "assessment", "penetration", "incident", "response", "disaster",
    "recovery", "continuity", "bcp", "drp", "gdpr", "hipaa", "soc2", "iso", "nist", "pci",
    "dss", "ccpa", "privacy", "protection", "grc", "cybersecurity", "information", "infosec",
    "threat", "intelligence", "soc", "event", "siem", "identity", "access", "iam", "privileged",
    "pam", "multi", "factor", "authentication", "mfa", "single", "sign", "on", "sso",
    "encryption", "decryption", "firewall", "ids", "ips", "vpn", "endpoint", "antivirus",
    "malware", "detection", "forensics", "handling", "assessments", "policies", "procedures",
    "guidelines", "mitre", "att&ck", "modeling", "secure", "lifecycle", "sdlc", "awareness",
    "phishing", "vishing", "smishing", "ransomware", "spyware", "adware", "rootkits",
    "botnets", "trojans", "viruses", "worms", "zero", "day", "exploits", "patches", "patching",
    "updates", "upgrades", "configuration", "ticketing", "crm", "erp", "scm", "hcm", "financial",
    "accounting", "bi", "warehousing", "etl", "extract", "transform", "load", "lineage",
    "master", "mdm", "lakes", "marts", "big", "hadoop", "spark", "kafka", "flink", "mongodb",
    "cassandra", "redis", "elasticsearch", "relational", "mysql", "postgresql", "db2",
    "teradata", "snowflake", "redshift", "synapse", "bigquery", "aurora", "dynamodb",
    "documentdb", "cosmosdb", "graph", "neo4j", "graphdb", "timeseries", "influxdb",
    "timescaledb", "columnar", "vertica", "clickhouse", "vector", "pinecone", "weaviate",
    "milvus", "qdrant", "chroma", "faiss", "annoy", "hnswlib", "scikit", "learn", "tensorflow",
    "pytorch", "keras", "xgboost", "lightgbm", "catboost", "statsmodels", "numpy", "pandas",
    "matplotlib", "seaborn", "plotly", "bokeh", "dash", "flask", "django", "fastapi", "spring",
    "boot", ".net", "core", "node.js", "express.js", "react", "angular", "vue.js", "svelte",
    "jquery", "bootstrap", "tailwind", "sass", "less", "webpack", "babel", "npm", "yarn",
    "ansible", "terraform", "jenkins", "gitlab", "github", "actions", "codebuild", "codepipeline",
    "codedeploy", "build", "deploy", "run", "lambda", "functions", "serverless", "microservices",
    "gateway", "mesh", "istio", "linkerd", "grpc", "restful", "soap", "message", "queues",
    "rabbitmq", "activemq", "bus", "sqs", "sns", "pubsub", "version", "control", "svn",
    "mercurial", "trello", "asana", "monday.com", "smartsheet", "project", "primavera",
    "zendesk", "freshdesk", "itil", "cobit", "prince2", "pmp", "master", "owner", "lean",
    "six", "sigma", "black", "belt", "green", "yellow", "qms", "9001", "27001", "14001",
    "ohsas", "18001", "sa", "8000", "cmii", "cmi", "cism", "cissp", "ceh", "comptia",
    "security+", "network+", "a+", "linux+", "ccna", "ccnp", "ccie", "certified", "solutions",
    "architect", "developer", "sysops", "administrator", "specialty", "professional", "azure",
    "az-900", "az-104", "az-204", "az-303", "az-304", "az-400", "az-500", "az-700", "az-800",
    "az-801", "dp-900", "dp-100", "dp-203", "ai-900", "ai-102", "da-100", "pl-900", "pl-100",
    "pl-200", "pl-300", "pl-400", "pl-500", "ms-900", "ms-100", "ms-101", "ms-203", "ms-500",
    "ms-700", "ms-720", "ms-740", "ms-600", "sc-900", "sc-200", "sc-300", "sc-400", "md-100",
    "md-101", "mb-200", "mb-210", "mb-220", "mb-230", "mb-240", "mb-260", "mb-300", "mb-310",
    "mb-320", "mb-330", "mb-340", "mb-400", "mb-500", "mb-600", "mb-700", "mb-800", "mb-910",
    "mb-920", "gcp-ace", "gcp-pca", "gcp-pde", "gcp-pse", "gcp-pml", "gcp-psa", "gcp-pcd",
    "gcp-pcn", "gcp-psd", "gcp-pda", "gcp-pci", "gcp-pws", "gcp-pwa", "gcp-pme", "gcp-pms",
    "gcp-pmd", "gcp-pma", "gcp-pmc", "gcp-pmg", "cisco", "juniper", "red", "hat", "rhcsa",
    "rhce", "vmware", "vcpa", "vcpd", "vcpi", "vcpe", "vcpx", "citrix", "cc-v", "cc-p",
    "cc-e", "cc-m", "cc-s", "cc-x", "palo", "alto", "pcnsa", "pcnse", "fortinet", "fcsa",
    "fcsp", "fcc", "fcnsp", "fct", "fcp", "fcs", "fce", "fcn", "fcnp", "fcnse"
])
STOP_WORDS = NLTK_STOP_WORDS.union(CUSTOM_STOP_WORDS)

SKILL_CATEGORIES = {
    "Programming Languages": ["Python", "Java", "JavaScript", "C++", "C#", "Go", "Ruby", "PHP", "Swift", "Kotlin", "TypeScript", "R", "Bash Scripting", "Shell Scripting"],
    "Web Technologies": ["HTML5", "CSS3", "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Spring Boot", "Express.js", "WebSockets"],
    "Databases": ["SQL", "NoSQL", "PostgreSQL", "MySQL", "MongoDB", "Cassandra", "Elasticsearch", "Neo4j", "Redis", "BigQuery", "Snowflake", "Redshift", "Aurora", "DynamoDB", "DocumentDB", "CosmosDB"],
    "Cloud Platforms": ["AWS", "Azure", "Google Cloud Platform", "GCP", "Serverless", "AWS Lambda", "Azure Functions", "Google Cloud Functions"],
    "DevOps & MLOps": ["Git", "GitHub", "GitLab", "Bitbucket", "CI/CD", "Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins", "CircleCI", "GitHub Actions", "Azure DevOps", "MLOps"],
    "Data Science & ML": ["Machine Learning", "Deep Learning", "Natural Language Processing", "Computer Vision", "Reinforcement Learning", "Scikit-learn", "TensorFlow", "PyTorch", "Keras", "XGBoost", "LightGBM", "Data Cleaning", "Feature Engineering",
    "Model Evaluation", "Statistical Modeling", "Time Series Analysis", "Predictive Modeling", "Clustering",
    "Classification", "Regression", "Neural Networks", "Convolutional Networks", "Recurrent Networks",
    "Transformers", "LLMs", "Prompt Engineering", "Generative AI", "MLOps", "Data Munging", "A/B Testing",
    "Experiment Design", "Hypothesis Testing", "Bayesian Statistics", "Causal Inference", "Graph Neural Networks"],
    "Data Analytics & BI": ["Data Cleaning", "Feature Engineering", "Model Evaluation", "Statistical Analysis", "Time Series Analysis", "Data Munging", "A/B Testing", "Experiment Design", "Hypothesis Testing", "Bayesian Statistics", "Causal Inference", "Excel (Advanced)", "Tableau", "Power BI", "Looker", "Qlik Sense", "Google Data Studio", "Dax", "M Query", "ETL", "ELT", "Data Warehousing", "Data Lake", "Data Modeling", "Business Intelligence", "Data Visualization", "Dashboarding", "Report Generation", "Google Analytics"],
    "Soft Skills": ["Stakeholder Management", "Risk Management", "Change Management", "Communication Skills", "Public Speaking", "Presentation Skills", "Cross-functional Collaboration",
    "Problem Solving", "Critical Thinking", "Analytical Skills", "Adaptability", "Time Management",
    "Organizational Skills", "Attention to Detail", "Leadership", "Mentorship", "Team Leadership",
    "Decision Making", "Negotiation", "Client Management", "Stakeholder Communication", "Active Listening",
    "Creativity", "Innovation", "Research", "Data Analysis", "Report Writing", "Documentation"],
    "Project Management": ["Agile Methodologies", "Scrum", "Kanban", "Jira", "Trello", "Product Lifecycle", "Sprint Planning", "Project Charter", "Gantt Charts", "MVP", "Backlog Grooming",
    "Program Management", "Portfolio Management", "PMP", "CSM"],
    "Security": ["Cybersecurity", "Information Security", "Risk Assessment", "Compliance", "GDPR", "HIPAA", "ISO 27001", "Penetration Testing", "Vulnerability Management", "Incident Response", "Security Audits", "Forensics", "Threat Intelligence", "SIEM", "Firewall Management", "Endpoint Security", "IAM", "Cryptography", "Network Security", "Application Security", "Cloud Security"],
    "Other Tools & Frameworks": ["Jira", "Confluence", "Swagger", "OpenAPI", "Zendesk", "ServiceNow", "Intercom", "Live Chat", "Ticketing Systems", "HubSpot", "Salesforce Marketing Cloud",
    "QuickBooks", "SAP FICO", "Oracle Financials", "Workday", "Microsoft Dynamics", "NetSuite", "Adobe Creative Suite", "Canva", "Mailchimp", "Hootsuite", "Buffer", "SEMrush", "Ahrefs", "Moz", "Screaming Frog",
    "JMeter", "Postman", "SoapUI", "SVN", "Perforce", "Asana", "Monday.com", "Miro", "Lucidchart", "Visio", "MS Project", "Primavera", "AutoCAD", "SolidWorks", "MATLAB", "LabVIEW", "Simulink", "ANSYS",
    "CATIA", "NX", "Revit", "ArcGIS", "QGIS", "OpenCV", "NLTK", "SpaCy", "Gensim", "Hugging Face Transformers",
    "Docker Compose", "Helm", "Ansible Tower", "SaltStack", "Chef InSpec", "Terraform Cloud", "Vault",
    "Consul", "Nomad", "Prometheus", "Grafana", "Alertmanager", "Loki", "Tempo", "Jaeger", "Zipkin",
    "Fluentd", "Logstash", "Kibana", "Grafana Loki", "Datadog", "New Relic", "AppDynamics", "Dynatrace",
    "Nagios", "Zabbix", "Icinga", "PRTG", "SolarWinds", "Wireshark", "Nmap", "Metasploit", "Burp Suite",
    "OWASP ZAP", "Nessus", "Qualys", "Rapid7", "Tenable", "CrowdStrike", "SentinelOne", "Palo Alto Networks",
    "Fortinet", "Cisco Umbrella", "Okta", "Auth0", "Keycloak", "Ping Identity", "Active Directory",
    "LDAP", "OAuth", "JWT", "OpenID Connect", "SAML", "MFA", "SSO", "PKI", "TLS/SSL", "VPN", "IDS/IPS",
    "DLP", "CASB", "SOAR", "XDR", "EDR", "MDR", "GRC", "ITIL", "Lean Six Sigma", "CFA", "CPA", "SHRM-CP",
    "PHR", "CEH", "OSCP", "CCNA", "CISSP", "CISM", "CompTIA Security+"]
}
MASTER_SKILLS = set([skill for category_list in SKILL_CATEGORIES.values() for skill in category_list])

# --- Resume Processing Helpers (from screener.py) ---
def clean_text(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip().lower()

def extract_relevant_keywords(text, filter_set):
    cleaned_text = clean_text(text)
    extracted_keywords = set()
    categorized_keywords = collections.defaultdict(list)
    if filter_set:
        sorted_filter_skills = sorted(list(filter_set), key=len, reverse=True)
        temp_text = cleaned_text
        for skill_phrase in sorted_filter_skills:
            pattern = r'\b' + re.escape(skill_phrase.lower()) + r'\b'
            matches = re.findall(pattern, temp_text)
            if matches:
                extracted_keywords.add(skill_phrase.lower())
                found_category = False
                for category, skills_in_category in SKILL_CATEGORIES.items():
                    if skill_phrase.lower() in [s.lower() for s in skills_in_category]:
                        categorized_keywords[category].append(skill_phrase)
                        found_category = True
                        break
                if not found_category:
                    categorized_keywords["Uncategorized"].append(skill_phrase)
                temp_text = re.sub(pattern, " ", temp_text)
        individual_words_remaining = set(re.findall(r'\b\w+\b', temp_text))
        for word in individual_words_remaining:
            if word in filter_set:
                extracted_keywords.add(word)
                found_category = False
                for category, skills_in_category in SKILL_CATEGORIES.items():
                    if word.lower() in [s.lower() for s in skills_in_category]:
                        categorized_keywords[category].append(word)
                        found_category = True
                        break
                if not found_category:
                    categorized_keywords["Uncategorized"].append(word)
    else:
        all_words = set(re.findall(r'\b\w+\b', cleaned_text))
        extracted_keywords = {word for word in all_words if word not in STOP_WORDS}
        for word in extracted_keywords:
            categorized_keywords["General Keywords"].append(word)
    return extracted_keywords, dict(categorized_keywords)

def extract_text_from_pdf(uploaded_file):
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            return ''.join(page.extract_text() or '' for page in pdf.pages)
    except Exception as e:
        return f"[ERROR] {str(e)}"

def extract_years_of_experience(text):
    text = text.lower()
    total_months = 0
    job_date_ranges = re.findall(
        r'(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})\s*(?:to|–|-)\s*(present|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})',
        text
    )
    for start, end in job_date_ranges:
        try:
            start_date = datetime.datetime.strptime(start.strip(), '%b %Y')
        except ValueError:
            try:
                start_date = datetime.datetime.strptime(start.strip(), '%B %Y')
            except ValueError:
                continue
        if end.strip() == 'present':
            end_date = datetime.datetime.now()
        else:
            try:
                end_date = datetime.datetime.strptime(end.strip(), '%b %Y')
            except ValueError:
                try:
                    end_date = datetime.datetime.strptime(end.strip(), '%B %Y')
                except ValueError:
                    continue
        delta_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        total_months += max(delta_months, 0)
    if total_months == 0:
        match = re.search(r'(\d+(?:\.\d+)?)\s*(\+)?\s*(year|yrs|years)\b', text)
        if not match:
            match = re.search(r'experience[^\d]{0,10}(\d+(?:\.\d+)?)', text)
        if match:
            return float(match.group(1))
    return round(total_months / 12, 1)

def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else None

def extract_phone_number(text):
    match = re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', text)
    return match.group(0) if match else None

def extract_location(text):
    found_locations = set()
    text_lower = text.lower()
    sorted_cities = sorted(list(MASTER_CITIES), key=len, reverse=True)
    for city in sorted_cities:
        pattern = r'\b' + re.escape(city.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found_locations.add(city)
    return ", ".join(sorted(list(found_locations))) if found_locations else "Not Found"

def extract_name(text):
    lines = text.strip().split('\n')
    if not lines: return None
    EXCLUDE_NAME_TERMS = {"linkedin", "github", "portfolio", "resume", "cv", "profile", "contact", "email", "phone"}
    potential_name_lines = []
    for line in lines[:5]:
        line = line.strip()
        line_lower = line.lower()
        if not re.search(r'[@\d\.\-]', line) and \
           len(line.split()) <= 4 and \
           not any(term in line_lower for term in EXCLUDE_NAME_TERMS):
            if line.isupper() or (line and line[0].isupper() and all(word[0].isupper() or not word.isalpha() for word in line.split())):
                potential_name_lines.append(line)
    if potential_name_lines:
        name = max(potential_name_lines, key=len)
        name = re.sub(r'summary|education|experience|skills|projects|certifications|profile|contact', '', name, flags=re.IGNORECASE).strip()
        name = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', name).strip()
        if name: return name.title()
    return None

def extract_cgpa(text):
    text = text.lower()
    matches = re.findall(r'(?:cgpa|gpa|grade point average)\s*[:\s]*(\d+\.\d+)(?:\s*[\/of]{1,4}\s*(\d+\.\d+|\d+))?|(\d+\.\d+)(?:\s*[\/of]{1,4}\s*(\d+\.\d+|\d+))?\s*(?:cgpa|gpa)', text)
    for match in matches:
        if match[0] and match[0].strip():
            raw_cgpa = float(match[0])
            scale = float(match[1]) if match[1] else None
        elif match[2] and match[2].strip():
            raw_cgpa = float(match[2])
            scale = float(match[3]) if match[3] else None
        else: continue
        if scale and scale not in [0, 1]:
            normalized_cgpa = (raw_cgpa / scale) * 4.0
            return round(normalized_cgpa, 2)
        elif raw_cgpa <= 4.0: return round(raw_cgpa, 2)
        elif raw_cgpa <= 10.0: return round((raw_cgpa / 10.0) * 4.0, 2)
    return None

def extract_education_details(text):
    education_section_matches = re.finditer(r'(?:education|academic background|qualifications)\s*(\n|$)', text, re.IGNORECASE)
    education_details = []
    start_index = -1
    for match in education_section_matches: start_index = match.end(); break
    if start_index != -1:
        sections = ['experience', 'work history', 'skills', 'projects', 'certifications', 'awards', 'publications']
        end_index = len(text)
        for section in sections:
            section_match = re.search(r'\b' + re.escape(section) + r'\b', text[start_index:], re.IGNORECASE)
            if section_match: end_index = start_index + section_match.start(); break
        education_text = text[start_index:end_index].strip()
        edu_blocks = re.split(r'\n(?=\s*(?:bachelor|master|phd|associate|diploma|certificat|graduat|postgraduat|doctorate|university|college|institute|school|academy)\b|\d{4}\s*[-–]\s*(?:\d{4}|present))', education_text, flags=re.IGNORECASE)
        for block in edu_blocks:
            block = block.strip()
            if not block: continue
            uni, degree, major, year = None, None, None, None
            year_match = re.search(r'(\d{4})\s*[-–]\s*(\d{4}|present)|\b(\d{4})\b', block)
            if year_match:
                if year_match.group(1) and year_match.group(2): year = f"{year_match.group(1)}-{year_match.group(2)}"
                elif year_match.group(3): year = year_match.group(3)
            degree_match = re.search(r'\b(b\.?s\.?|bachelor of science|b\.?a\.?|bachelor of arts|m\.?s\.?|master of science|m\.?a\.?|master of arts|ph\.?d\.?|doctor of philosophy|mba|master of business administration|diploma|certificate)\b', block, re.IGNORECASE)
            if degree_match: degree = degree_match.group(0).title()
            uni_match = re.search(r'\b([A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*)\s+(?:university|college|institute|school|academy)\b', block, re.IGNORECASE)
            if uni_match: uni = uni_match.group(1)
            else:
                lines = block.split('\n')
                for line in lines:
                    potential_uni_match = re.search(r'^[A-Z][a-zA-Z\s,&\.]+\b(university|college|institute|school|academy)?', line.strip())
                    if potential_uni_match and len(potential_uni_match.group(0).split()) > 1: uni = potential_uni_match.group(0).strip().replace(',', ''); break
            major_match = re.search(r'(?:in|of)\s+([A-Z][a-zA-Z\s]+(?:engineering|science|arts|business|management|studies|technology))', block, re.IGNORECASE)
            if major_match: major = major_match.group(1).strip()
            if uni or degree or major or year: education_details.append({"University": uni, "Degree": degree, "Major": major, "Year": year})
    return education_details

def extract_work_history(text):
    work_history_section_matches = re.finditer(r'(?:experience|work history|employment history)\s*(\n|$)', text, re.IGNORECASE)
    work_details = []
    start_index = -1
    for match in work_history_section_matches: start_index = match.end(); break
    if start_index != -1:
        sections = ['education', 'skills', 'projects', 'certifications', 'awards', 'publications']
        end_index = len(text)
        for section in sections:
            section_match = re.search(r'\b' + re.escape(section) + r'\b', text[start_index:], re.IGNORECASE)
            if section_match: end_index = start_index + section_match.start(); break
        work_text = text[start_index:end_index].strip()
        job_blocks = re.split(r'\n(?=[A-Z][a-zA-Z\s,&\.]+(?:\s(?:at|@))?\s*[A-Z][a-zA-Z\s,&\.]*\s*(?:-|\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}))', work_text, flags=re.IGNORECASE)
        for block in job_blocks:
            block = block.strip()
            if not block: continue
            company, title, start_date, end_date = None, None, None, None
            date_range_match = re.search(
                r'((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}|\d{4})\s*[-–]\s*(present|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}|\d{4})',
                block, re.IGNORECASE
            )
            if date_range_match:
                start_date = date_range_match.group(1)
                end_date = date_range_match.group(2)
                block = block.replace(date_range_match.group(0), '').strip()
            lines = block.split('\n')
            for line in lines:
                line = line.strip()
                if not line: continue
                title_company_match = re.search(r'([A-Z][a-zA-Z\s,\-&.]+)\s+(?:at|@)\s+([A-Z][a-zA-Z\s,\-&.]+)', line)
                if title_company_match: title, company = title_company_match.group(1).strip(), title_company_match.group(2).strip(); break
                company_title_match = re.search(r'^([A-Z][a-zA-Z\s,\-&.]+),\s*([A-Z][a-zA-Z\s,\-&.]+)', line)
                if company_title_match: company, title = company_title_match.group(1).strip(), company_title_match.group(2).strip(); break
                if not company and not title:
                    potential_org_match = re.search(r'^[A-Z][a-zA-Z\s,\-&.]+', line)
                    if potential_org_match and len(potential_org_match.group(0).split()) > 1:
                        if not company: company = potential_org_match.group(0).strip()
                        elif not title: title = potential_org_match.group(0).strip()
                        break
            if company or title or start_date or end_date: work_details.append({"Company": company, "Title": title, "Start Date": start_date, "End Date": end_date})
    return work_details

def extract_project_details(text):
    project_section_matches = re.finditer(r'(?:projects|personal projects|key projects)\s*(\n|$)', text, re.IGNORECASE)
    project_details = []
    start_index = -1
    for match in project_section_matches: start_index = match.end(); break
    if start_index != -1:
        sections = ['education', 'experience', 'work history', 'skills', 'certifications', 'awards', 'publications']
        end_index = len(text)
        for section in sections:
            section_match = re.search(r'\b' + re.escape(section) + r'\b', text[start_index:], re.IGNORECASE)
            if section_match: end_index = start_index + section_match.start(); break
        project_text = text[start_index:end_index].strip()
        project_blocks = re.split(r'\n(?=[A-Z][a-zA-Z\s,&\-]+\s*(?:\()?\d{4}(?:\))?)|\n(?=•\s*[A-Z][a-zA-Z\s,&\-]+)', project_text)
        for block in project_blocks:
            block = block.strip()
            if not block: continue
            title, description, technologies = None, [], []
            lines = block.split('\n')
            if lines:
                title_line = lines[0].strip()
                if len(title_line.split()) <= 10 and re.match(r'^[A-Z]', title_line): title, description_lines = title_line, lines[1:]
                else: description_lines = lines
                block_lower = block.lower()
                for skill in MASTER_SKILLS:
                    if re.search(r'\b' + re.escape(skill.lower()) + r'\b', block_lower): technologies.append(skill)
                description = [line.strip() for line in description_lines if line.strip()]
            if title or description or technologies: project_details.append({"Project Title": title, "Description": "\n".join(description), "Technologies Used": ", ".join(technologies)})
    return project_details

def extract_languages(text):
    languages_list = []
    text_lower = text.lower()
    all_languages = [
        "english", "hindi", "spanish", "french", "german", "mandarin", "japanese", "arabic",
        "russian", "portuguese", "italian", "korean", "bengali", "marathi", "telugu", "tamil",
        "gujarati", "urdu", "kannada", "odia", "malayalam", "punjabi", "assamese", "kashmiri",
        "sindhi", "sanskrit", "dutch", "swedish", "norwegian", "danish", "finnish", "greek",
        "turkish", "hebrew", "thai", "vietnamese", "indonesian", "malay", "filipino", "swahili",
        "farsi", "persian", "polish", "ukrainian", "romanian", "czech", "slovak", "hungarian"
    ]
    languages_section_match = re.search(r'\b(languages|language skills|linguistic abilities)\b\s*(\n|$)', text_lower)
    if languages_section_match:
        start_index = languages_section_match.end()
        sections = ['education', 'experience', 'work history', 'skills', 'projects', 'certifications', 'awards', 'publications', 'interests', 'hobbies']
        end_index = len(text_lower)
        for section in sections:
            section_match = re.search(r'\b' + re.escape(section) + r'\b', text_lower[start_index:], re.IGNORECASE)
            if section_match: end_index = start_index + section_match.start(); break
        languages_text = text_lower[start_index:end_index].strip()
        for lang in all_languages:
            if re.search(r'\b' + re.escape(lang) + r'\b', languages_text): languages_list.append(lang.title())
    else:
        for lang in all_languages:
            if re.search(r'\b' + re.escape(lang) + r'\b', text_lower):
                if lang.title() not in languages_list: languages_list.append(lang.title())
    return ", ".join(sorted(list(set(languages_list)))) if languages_list else "Not Found"

def format_education_details(edu_list):
    if not edu_list: return "Not Found"
    formatted_entries = []
    for entry in edu_list:
        parts = []
        if entry.get("Degree"): parts.append(entry["Degree"])
        if entry.get("Major"): parts.append(f"in {entry['Major']}")
        if entry.get("University"): parts.append(f"from {entry['University']}")
        if entry.get("Year"): parts.append(f"({entry['Year']})")
        formatted_entries.append(" ".join(parts).strip())
    return "; ".join(formatted_entries) if formatted_entries else "Not Found"

def format_work_history(work_list):
    if not work_list: return "Not Found"
    formatted_entries = []
    for entry in work_list:
        parts = []
        if entry.get("Title"): parts.append(entry["Title"])
        if entry.get("Company"): parts.append(f"at {entry['Company']}")
        if entry.get("Start Date") and entry.get("End Date"): parts.append(f"({entry['Start Date']} - {entry['End Date']})")
        elif entry.get("Start Date"): parts.append(f"(Since {entry['Start Date']})")
        formatted_entries.append(" ".join(parts).strip())
    return "; ".join(formatted_entries) if formatted_entries else "Not Found"

def format_project_details(proj_list):
    if not proj_list: return "Not Found"
    formatted_entries = []
    for entry in proj_list:
        parts = []
        if entry.get("Project Title"): parts.append(f"**{entry['Project Title']}**")
        if entry.get("Technologies Used"): parts.append(f"({entry['Technologies Used']})")
        if entry.get("Description") and entry["Description"].strip():
            desc_snippet = entry["Description"].split('\n')[0][:50] + "..." if len(entry["Description"]) > 50 else entry["Description"]
            parts.append(f'"{desc_snippet}"')
        formatted_entries.append(" ".join(parts).strip())
    return "; ".join(formatted_entries) if formatted_entries else "Not Found"

@st.cache_data(show_spinner="Generating concise AI Suggestion...")
def generate_concise_ai_suggestion(candidate_name, score, years_exp, semantic_similarity, cgpa):
    overall_fit_description = ""
    review_focus_text = ""
    key_strength_hint = ""
    high_score, moderate_score, high_exp, moderate_exp, high_sem_sim, moderate_sem_sim, high_cgpa, moderate_cgpa = 85, 65, 4, 2, 0.75, 0.4, 3.5, 3.0
    if score >= high_score and years_exp >= high_exp and semantic_similarity >= high_sem_sim:
        overall_fit_description = "High alignment."
        key_strength_hint = "Strong technical and experience match, quick integration expected."
        review_focus_text = "Cultural fit, project contributions."
    elif score >= moderate_score and years_exp >= moderate_exp and semantic_similarity >= moderate_sem_sim:
        overall_fit_description = "Moderate fit."
        key_strength_hint = "Good foundational skills, potential for growth."
        review_focus_text = "Depth of experience, skill application, learning agility."
    else:
        overall_fit_description = "Limited alignment."
        key_strength_hint = "May require significant development or a different role."
        review_focus_text = "Foundational skills, transferable experience, long-term potential."
    cgpa_note = ""
    if cgpa is not None:
        if cgpa >= high_cgpa: cgpa_note = "Excellent academic record. "
        elif cgpa >= moderate_cgpa: cgpa_note = "Solid academic background. "
        else: cgpa_note = "Academic record may need review. "
    else: cgpa_note = "CGPA not found. "
    summary_text = f"**Fit:** {overall_fit_description} **Strengths:** {cgpa_note}{key_strength_hint} **Focus:** {review_focus_text}"
    return summary_text

@st.cache_data(show_spinner="Generating detailed HR Assessment...")
def generate_detailed_hr_assessment(candidate_name, score, years_exp, semantic_similarity, cgpa, jd_text, resume_text, matched_keywords, missing_skills, max_exp_cutoff):
    assessment_parts = []
    overall_assessment_title = ""
    matched_kws_str = ", ".join(matched_keywords) if isinstance(matched_keywords, list) else matched_keywords
    missing_skills_str = ", ".join(missing_skills) if isinstance(missing_skills, list) else missing_skills
    high_score, strong_score, promising_score = 90, 80, 60
    high_exp, strong_exp, promising_exp = 5, 3, 1
    high_sem_sim, strong_sem_sim, promising_sem_sim = 0.85, 0.7, 0.35
    high_cgpa, strong_cgpa, promising_cgpa = 3.5, 3.0, 2.5

    if score >= high_score and years_exp >= high_exp and years_exp <= max_exp_cutoff and semantic_similarity >= high_sem_sim and (cgpa is None or cgpa >= high_cgpa):
        overall_assessment_title = "Exceptional Candidate: Highly Aligned with Strategic Needs"
        assessment_parts.append(f"**{candidate_name}** presents an **exceptional profile** with a high score of {score:.2f}% and {years_exp:.1f} years of experience. This demonstrates a profound alignment with the job description's core requirements, further evidenced by a strong semantic similarity of {semantic_similarity:.2f}.")
        if cgpa is not None: assessment_parts.append(f"Their academic record, with a CGPA of {cgpa:.2f} (normalized to 4.0 scale), further solidifies their strong foundational knowledge.")
        assessment_parts.append(f"**Key Strengths:** This candidate possesses a robust skill set directly matching critical keywords in the JD, including: *{matched_kws_str if matched_kws_str else 'No specific keywords listed, but overall strong match'}*. Their extensive experience indicates a capacity for leadership and handling complex challenges, suggesting immediate productivity and minimal ramp-up time. They are poised to make significant contributions from day one.")
        assessment_parts.append("The resume highlights a clear career progression and a history of successful project delivery, often exceeding expectations. Their qualifications exceed expectations, making them a top-tier applicant for this role.")
        assessment_parts.append("This individual's profile suggests they are not only capable of fulfilling the role's duties but also have the potential to mentor others, drive innovation, and take on strategic initiatives within the team. Their background indicates a strong fit for a high-impact position.")
        assessment_parts.append(f"**Action:** Strongly recommend for immediate interview. Prioritize for hiring and consider for advanced roles if applicable.")
    elif score >= strong_score and years_exp >= strong_exp and years_exp <= max_exp_cutoff and semantic_similarity >= strong_sem_sim and (cgpa is None or cgpa >= strong_cgpa):
        overall_assessment_title = "Strong Candidate: Excellent Potential for Key Contributions"
        assessment_parts.append(f"**{candidate_name}** is a **strong candidate** with a score of {score:.2f}% and {years_exp:.1f} years of experience. They show excellent alignment with the job description, supported by a solid semantic similarity of {semantic_similarity:.2f}.")
        if cgpa is not None: assessment_parts.append(f"Their academic performance, with a CGPA of {cgpa:.2f}, indicates a solid theoretical grounding.")
        assessment_parts.append(f"**Key Strengths:** Significant overlap in required skills and practical experience that directly addresses the job's demands. Matched keywords include: *{matched_kws_str if matched_kws_str else 'No specific keywords listed, but overall strong match'}*. This individual is likely to integrate well and contribute effectively from an early stage, bringing valuable expertise to the team.")
        assessment_parts.append("Their resume indicates a consistent track record of achieving results and adapting to new challenges. They demonstrate a solid understanding of the domain and could quickly become a valuable asset, requiring moderate onboarding.")
        assessment_parts.append("This candidate is well-suited for the role and demonstrates the core competencies required. Their experience suggests they can handle typical challenges and contribute positively to team dynamics.")
        assessment_parts.append(f"**Action:** Recommend for interview. Good fit for the role, with potential for growth.")
    elif score >= promising_score and years_exp >= promising_exp and years_exp <= max_exp_cutoff and semantic_similarity >= promising_sem_sim and (cgpa is None or cgpa >= promising_cgpa):
        overall_assessment_title = "Promising Candidate: Requires Focused Review on Specific Gaps"
        assessment_parts.append(f"**{candidate_name}** is a **promising candidate** with a score of {score:.2f}% and {years_exp:.1f} years of experience. While demonstrating a foundational understanding (semantic similarity: {semantic_similarity:.2f}), there are areas that warrant deeper investigation to ensure a complete fit.")
        gaps_identified = []
        if score < 70: gaps_identified.append("The overall score suggests some core skill areas may need development or further clarification.")
        if years_exp < promising_exp: gaps_identified.append(f"Experience ({years_exp:.1f} yrs) is on the lower side; assess their ability to scale up quickly and take on more responsibility.")
        if semantic_similarity < 0.5: gaps_identified.append("Semantic understanding of the JD's nuances might be limited; probe their theoretical knowledge versus practical application in real-world scenarios.")
        if cgpa is not None and cgpa < promising_cgpa: gaps_identified.append(f"Academic record (CGPA: {cgpa:.2f}) is below preferred, consider its relevance to role demands.")
        if missing_skills_str: gaps_identified.append(f"**Potential Missing Skills:** *{missing_skills_str}*. Focus interview questions on these areas to assess their current proficiency or learning agility.")
        if years_exp > max_exp_cutoff: gaps_identified.append(f"Experience ({years_exp:.1f} yrs) exceeds the maximum desired ({max_exp_cutoff} yrs). Evaluate if this indicates overqualification or a potential mismatch in role expectations.")
        if gaps_identified: assessment_parts.append("Areas for further exploration include: " + " ".join(gaps_identified))
        assessment_parts.append("The candidate shows potential, especially if they can demonstrate quick learning or relevant transferable skills. Their resume indicates a willingness to grow and take on new challenges, which is a positive sign for development opportunities.")
        assessment_parts.append(f"**Action:** Consider for initial phone screen or junior role. Requires careful evaluation and potentially a development plan.")
    else:
        overall_assessment_title = "Limited Match: Consider Only for Niche Needs or Pipeline Building"
        assessment_parts.append(f"**{candidate_name}** shows a **limited match** with a score of {score:.2f}% and {years_exp:.1f} years of experience (semantic similarity: {semantic_similarity:.2f}). This profile indicates a significant deviation from the core requirements of the job description.")
        if cgpa is not None: assessment_parts.append(f"Their academic record (CGPA: {cgpa:.2f}) also indicates a potential mismatch.")
        assessment_parts.append(f"**Key Concerns:** A low overlap in essential skills and potentially insufficient experience for the role's demands. Many key skills appear to be missing: *{missing_skills_str if missing_skills_str else 'No specific missing skills listed, but overall low match'}*. While some transferable skills may exist, a substantial investment in training or a re-evaluation of role fit would likely be required for this candidate to succeed.")
        if years_exp > max_exp_cutoff: assessment_parts.append(f"Additionally, their experience ({years_exp:.1f} yrs) significantly exceeds the maximum desired ({max_exp_cutoff} yrs), which might indicate overqualification or a mismatch in career trajectory for this specific opening.")
        assessment_parts.append("The resume does not strongly align with the technical or experience demands of this specific position. Their background may be more suited for a different type of role or industry, or an entry-level position if their core skills are strong but experience is lacking.")
        assessment_parts.append("This candidate might not be able to meet immediate role requirements without extensive support. Their current profile suggests a mismatch with the current opening.")
        assessment_parts.append(f"**Action:** Not recommended for this role. Consider for other open positions or future pipeline, or politely decline.")
    final_assessment = f"**Overall HR Assessment: {overall_assessment_title}**\n\n"
    final_assessment += "\n".join(assessment_parts)
    return final_assessment

def semantic_score(resume_text, jd_text, years_exp, cgpa, high_priority_skills, medium_priority_skills, jd_skills_filter_set):
    jd_clean = clean_text(jd_text)
    resume_clean = clean_text(resume_text)
    score = 0.0
    semantic_similarity = 0.0
    resume_raw_skills, _ = extract_relevant_keywords(resume_clean, MASTER_SKILLS)
    jd_raw_skills, _ = extract_relevant_keywords(jd_clean, jd_skills_filter_set)
    weighted_keyword_overlap_score = 0
    total_jd_skill_weight = 0
    WEIGHT_HIGH, WEIGHT_MEDIUM, WEIGHT_BASE = 3, 2, 1
    for jd_skill in jd_raw_skills:
        current_weight = WEIGHT_BASE
        if jd_skill in [s.lower() for s in high_priority_skills]: current_weight = WEIGHT_HIGH
        elif jd_skill in [s.lower() for s in medium_priority_skills]: current_weight = WEIGHT_MEDIUM
        total_jd_skill_weight += current_weight
        if jd_skill in resume_raw_skills: weighted_keyword_overlap_score += current_weight
    if total_jd_skill_weight > 0: weighted_jd_coverage_percentage = (weighted_keyword_overlap_score / total_jd_skill_weight) * 100
    else: weighted_jd_coverage_percentage = 0.0

    if ml_model is None or model is None:
        st.warning("ML models not loaded. Providing basic score and generic feedback.")
        basic_score = (weighted_jd_coverage_percentage * 0.7)
        basic_score += min(years_exp * 5, 30)
        if cgpa is not None:
            if cgpa >= 3.5: basic_score += 5
            elif cgpa < 2.5: basic_score -= 5
        score = round(min(basic_score, 100), 2)
        return score, round(semantic_similarity, 2)

    try:
        jd_embed = model.encode(jd_clean)
        resume_embed = model.encode(resume_clean)
        semantic_similarity = cosine_similarity(jd_embed.reshape(1, -1), resume_embed.reshape(1, -1))[0][0]
        semantic_similarity = float(np.clip(semantic_similarity, 0, 1))
        years_exp_for_model = float(years_exp) if years_exp is not None else 0.0
        features = np.concatenate([jd_embed, resume_embed, [years_exp_for_model], [weighted_keyword_overlap_score]])
        predicted_score = ml_model.predict([features])[0]
        blended_score = (predicted_score * 0.6) + (weighted_jd_coverage_percentage * 0.1) + (semantic_similarity * 100 * 0.3)
        if semantic_similarity > 0.7 and years_exp >= 3: blended_score += 5
        if cgpa is not None:
            if cgpa >= 3.5: blended_score += 3
            elif cgpa >= 3.0: blended_score += 1
            elif cgpa < 2.5: blended_score -= 2
        score = float(np.clip(blended_score, 0, 100))
        return round(score, 2), round(semantic_similarity, 2)
    except Exception as e:
        st.warning(f"Error during semantic scoring, falling back to basic: {e}")
        basic_score = (weighted_jd_coverage_percentage * 0.7)
        basic_score += min(years_exp * 5, 30)
        if cgpa is not None:
            if cgpa >= 3.5: basic_score += 5
            elif cgpa < 2.5: basic_score -= 5
        score = round(min(basic_score, 100), 2)
        return score, 0.0

def create_mailto_link(recipient_email, candidate_name, job_title="Job Opportunity", sender_name="Recruiting Team"):
    subject = urllib.parse.quote(f"Invitation for Interview - {job_title} - {candidate_name}")
    body = urllib.parse.quote(f"""Dear {candidate_name},

We were very impressed with your profile and would like to invite you for an interview for the {job_title} position.

Best regards,

The {sender_name}""")
    return f"mailto:{recipient_email}?subject={subject}&body={body}"

# --- Resume Screener Page (from screener.py) ---
def resume_screener_page():
    st.title("🧠 ScreenerPro – AI-Powered Resume Screener")

    # Define all expected columns for the DataFrame.
    expected_cols = [
        "File Name", "Candidate Name", "Score (%)", "Years Experience", "CGPA (4.0 Scale)",
        "Email", "Phone Number", "Location", "Languages", "Education Details",
        "Work History", "Project Details", "AI Suggestion", "Detailed HR Assessment",
        "Matched Keywords", "Missing Skills", "Matched Keywords (Categorized)",
        "Missing Skills (Categorized)", "Semantic Similarity", "Resume Raw Text",
        "JD Used", "Shortlisted", "Notes", "Tag"
    ]

    # Initialize full_results_df in session state if it doesn't exist or is not a DataFrame
    if 'full_results_df' not in st.session_state or not isinstance(st.session_state['full_results_df'], pd.DataFrame):
        st.session_state['full_results_df'] = pd.DataFrame(columns=expected_cols)
        st.session_state['full_results_df']['Shortlisted'] = st.session_state['full_results_df']['Shortlisted'].astype(bool)
        st.session_state['full_results_df']['Notes'] = st.session_state['full_results_df']['Notes'].astype(str)
        st.session_state['full_results_df']['Tag'] = st.session_state['full_results_df']['Tag'].astype(str)
    elif st.session_state['full_results_df'].empty:
        if not all(col in st.session_state['full_results_df'].columns for col in expected_cols):
            st.session_state['full_results_df'] = pd.DataFrame(columns=expected_cols)
            st.session_state['full_results_df']['Shortlisted'] = st.session_state['full_results_df']['Shortlisted'].astype(bool)
            st.session_state['full_results_df']['Notes'] = st.session_state['full_results_df']['Notes'].astype(str)
            st.session_state['full_results_df']['Tag'] = st.session_state['full_results_df']['Tag'].astype(str)

    # --- Job Description and Controls Section ---
    st.markdown("## ⚙️ Define Job Requirements & Screening Criteria")
    col1, col2 = st.columns([2, 1])

    with col1:
        jd_text = ""
        job_roles = {"Upload my own": None}
        if not os.path.exists("data"):
            os.makedirs("data") # Ensure 'data' directory exists
        for fname in os.listdir("data"):
            if fname.endswith(".txt"):
                job_roles[fname.replace(".txt", "").replace("_", " ").title()] = os.path.join("data", fname)

        jd_option = st.selectbox("📌 **Select a Pre-Loaded Job Role or Upload Your Own Job Description**", list(job_roles.keys()))
        
        jd_name_for_results = ""
        if jd_option == "Upload my own":
            jd_file = st.file_uploader("Upload Job Description (TXT)", type="txt", help="Upload a .txt file containing the job description.")
            if jd_file:
                jd_text = jd_file.read().decode("utf-8")
                jd_name_for_results = jd_file.name.replace(".txt", "")
            else:
                jd_name_for_results = "Uploaded JD (No file selected)"
        else:
            jd_path = job_roles[jd_option]
            if jd_path and os.path.exists(jd_path):
                with open(jd_path, "r", encoding="utf-8") as f:
                    jd_text = f.read()
            jd_name_for_results = jd_option

        if jd_text:
            with st.expander("📝 View Loaded Job Description"):
                st.text_area("Job Description Content", jd_text, height=200, disabled=True, label_visibility="collapsed")

    with col2:
        cutoff = st.slider("📈 **Minimum Score Cutoff (%)**", 0, 100, 75, help="Candidates scoring below this percentage will be flagged for closer review or considered less suitable.")
        st.session_state['screening_cutoff_score'] = cutoff

        min_experience = st.slider("💼 **Minimum Experience Required (Years)**", 0, 15, 2, help="Candidates with less than this experience will be noted.")
        st.session_state['screening_min_experience'] = min_experience

        max_experience = st.slider("⬆️ **Maximum Experience Allowed (Years)**", 0, 20, 10, help="Candidates with more than this experience might be considered overqualified or outside the target range.")
        st.session_state['screening_max_experience'] = max_experience

        min_cgpa = st.slider("🎓 **Minimum CGPA Required (4.0 Scale)**", 0.0, 4.0, 2.5, 0.1, help="Candidates with CGPA below this value (normalized to 4.0) will be noted.")
        st.session_state['screening_min_cgpa'] = min_cgpa

        st.markdown("---")
        st.info("Once criteria are set, upload resumes below to begin screening.")

    # --- Skill Weighting Section ---
    st.markdown("## 🎯 Skill Prioritization (Optional)")
    st.caption("Assign higher importance to specific skills in the Job Description.")
    
    all_master_skills = sorted(list(MASTER_SKILLS))

    col_weights_1, col_weights_2 = st.columns(2)
    with col_weights_1:
        high_priority_skills = st.multiselect(
            "🌟 **High Priority Skills (Weight x3)**",
            options=all_master_skills,
            help="Select skills that are absolutely critical for this role. These will significantly boost the score if found."
        )
    with col_weights_2:
        medium_priority_skills = st.multiselect(
            "✨ **Medium Priority Skills (Weight x2)**",
            options=[s for s in all_master_skills if s not in high_priority_skills],
            help="Select skills that are very important, but not as critical as high priority ones."
        )

    custom_jd_skills_input = st.text_input(
        "➕ **Add Custom JD Skills (comma-separated)**",
        help="Enter additional skills relevant to the Job Description, e.g., 'Leadership, Cloud Security, Microservices Architecture'. These will be considered in scoring."
    )
    custom_jd_skills_set = {s.strip().lower() for s in custom_jd_skills_input.split(',') if s.strip()}
    
    jd_skills_for_processing = MASTER_SKILLS.union(custom_jd_skills_set)

    resume_files = st.file_uploader("📄 **Upload Resumes (PDF)**", type="pdf", accept_multiple_files=True, help="Upload one or more PDF resumes for screening.")

    if jd_text and resume_files:
        st.markdown("---")
        st.markdown("## ☁️ Job Description Keyword Cloud")
        st.caption("Visualizing the most frequent and important keywords from the Job Description.")
        st.info("💡 To filter candidates by these skills, use the 'Filter Candidates by Skill' section below the main results table.")
        
        jd_words_for_cloud_set, _ = extract_relevant_keywords(jd_text, jd_skills_for_processing)
        jd_words_for_cloud = " ".join(list(jd_words_for_cloud_set))

        if jd_words_for_cloud:
            wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(jd_words_for_cloud)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No significant keywords to display for the Job Description. Please ensure your JD has sufficient content or adjust your SKILL_CATEGORIES list.")
        st.markdown("---")

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, file in enumerate(resume_files):
            status_text.text(f"Processing {file.name} ({i+1}/{len(resume_files)})...")
            progress_bar.progress((i + 1) / len(resume_files))

            text = extract_text_from_pdf(file)
            if text.startswith("[ERROR]"):
                st.error(f"Failed to process {file.name}: {text.replace('[ERROR] ', '')}")
                continue

            exp = extract_years_of_experience(text)
            email = extract_email(text)
            phone = extract_phone_number(text)
            location = extract_location(text)
            languages = extract_languages(text)
            
            education_details_formatted = format_education_details(extract_education_details(text))
            work_history_formatted = format_work_history(extract_work_history(text))
            project_details_formatted = format_project_details(extract_project_details(text))

            candidate_name = extract_name(text) or file.name.replace('.pdf', '').replace('_', ' ').title()
            cgpa = extract_cgpa(text)

            resume_raw_skills_set, resume_categorized_skills = extract_relevant_keywords(text, all_master_skills)
            jd_raw_skills_set, jd_categorized_skills = extract_relevant_keywords(jd_text, jd_skills_for_processing)

            matched_keywords = list(resume_raw_skills_set.intersection(jd_raw_skills_set))
            missing_skills = list(jd_raw_skills_set.difference(resume_raw_skills_set)) 

            score, semantic_similarity = semantic_score(text, jd_text, exp, cgpa, high_priority_skills, medium_priority_skills, jd_skills_for_processing)
            
            concise_ai_suggestion = generate_concise_ai_suggestion(
                candidate_name=candidate_name, score=score, years_exp=exp, semantic_similarity=semantic_similarity, cgpa=cgpa
            )
            detailed_hr_assessment = generate_detailed_hr_assessment(
                candidate_name=candidate_name, score=score, years_exp=exp, semantic_similarity=semantic_similarity,
                cgpa=cgpa, jd_text=jd_text, resume_text=text, matched_keywords=matched_keywords,
                missing_skills=missing_skills, max_exp_cutoff=max_experience
            )

            results.append({
                "File Name": file.name,
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
                "Shortlisted": False,
                "Notes": "",
                "Tag": ""
            })
        
        progress_bar.empty()
        status_text.empty()

        newly_processed_df = pd.DataFrame(results, columns=expected_cols)

        if not newly_processed_df.empty:
            # Merge new results with existing state, preserving user edits for 'Shortlisted' and 'Notes'
            # Use 'File Name' as the key to merge
            
            # 1. Take the current full_results_df from session state
            current_full_df = st.session_state['full_results_df'].copy()

            # 2. Identify files that are in both the new batch and the current session state
            common_files = pd.merge(
                current_full_df[['File Name', 'Shortlisted', 'Notes']],
                newly_processed_df[['File Name']],
                on='File Name',
                how='inner'
            )
            
            # 3. Drop common files from newly_processed_df to avoid duplicates during concat
            newly_processed_df_unique = newly_processed_df[
                ~newly_processed_df['File Name'].isin(current_full_df['File Name'])
            ]

            # 4. Concatenate the unique new files with the existing full_results_df
            # This adds genuinely new files
            st.session_state['full_results_df'] = pd.concat([current_full_df, newly_processed_df_unique], ignore_index=True)

            # 5. Update the screening data for common files (preserving Shortlisted/Notes)
            # Iterate through common files and update the corresponding rows in full_results_df
            # This is less efficient than a merge, but ensures specific columns are preserved.
            for _, row_common in common_files.iterrows():
                file_name = row_common['File Name']
                # Get the new screening data for this file
                new_screening_data = newly_processed_df[newly_processed_df['File Name'] == file_name].iloc[0]
                
                # Update all columns except 'Shortlisted' and 'Notes' in the session state DataFrame
                for col in expected_cols:
                    if col not in ['Shortlisted', 'Notes', 'File Name']:
                        st.session_state['full_results_df'].loc[
                            st.session_state['full_results_df']['File Name'] == file_name, col
                        ] = new_screening_data[col]

            # Ensure correct dtypes after all operations
            st.session_state['full_results_df']['Shortlisted'] = st.session_state['full_results_df']['Shortlisted'].astype(bool)
            st.session_state['full_results_df']['Notes'] = st.session_state['full_results_df']['Notes'].astype(str)
            st.session_state['full_results_df']['Tag'] = st.session_state['full_results_df']['Tag'].astype(str)

            st.session_state['full_results_df'] = st.session_state['full_results_df'].sort_values(by="Score (%)", ascending=False).reset_index(drop=True)
        else:
            st.info("No resumes were successfully processed. Please check your uploaded files.")
        
        st.session_state['full_results_df'].to_csv("results.csv", index=False)

        st.markdown("## 📊 Candidate Score Comparison")
        st.caption("Visual overview of how each candidate ranks against the job requirements.")
        if not st.session_state['full_results_df'].empty:
            fig, ax = plt.subplots(figsize=(12, 7))
            colors = ['#4CAF50' if s >= cutoff else '#FFC107' if s >= (cutoff * 0.75) else '#F44346' for s in st.session_state['full_results_df']['Score (%)']]
            bars = ax.bar(st.session_state['full_results_df']['Candidate Name'], st.session_state['full_results_df']['Score (%)'], color=colors)
            ax.set_xlabel("Candidate", fontsize=14)
            ax.set_ylabel("Score (%)", fontsize=14)
            ax.set_title("Resume Screening Scores Across Candidates", fontsize=16, fontweight='bold')
            ax.set_ylim(0, 100)
            plt.xticks(rotation=60, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}", ha='center', va='bottom', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Upload resumes to see a comparison chart.")

        st.markdown("---")

        st.markdown("## 👑 Top Candidate AI Assessment")
        st.caption("A concise, AI-powered assessment for the most suitable candidate.")
        
        if not st.session_state['full_results_df'].empty:
            top_candidate = st.session_state['full_results_df'].iloc[0]
            
            cgpa_display = f"{top_candidate['CGPA (4.0 Scale)']:.2f}" if top_candidate['CGPA (4.0 Scale)'] is not None else "N/A"
            semantic_sim_display = f"{top_candidate['Semantic Similarity']:.2f}" if top_candidate['Semantic Similarity'] is not None else "N/A"

            st.markdown(f"### **{top_candidate['Candidate Name']}**")
            st.markdown(f"**Score:** {top_candidate['Score (%)']:.2f}% | **Experience:** {top_candidate['Years Experience']:.1f} years | **CGPA:** {cgpa_display} (4.0 Scale) | **Semantic Similarity:** {semantic_sim_display}")
            st.markdown(f"**AI Assessment:**")
            st.markdown(top_candidate['Detailed HR Assessment'])
            
            st.markdown("#### Matched Skills Breakdown:")
            if top_candidate['Matched Keywords (Categorized)']:
                for category, skills in top_candidate['Matched Keywords (Categorized)'].items():
                    st.write(f"**{category}:** {', '.join(skills)}")
            else:
                st.write("No categorized matched skills found.")

            st.markdown("#### Missing Skills Breakdown (from JD):")
            if top_candidate['Missing Skills'].strip():
                missing_skills_list = [s.strip() for s in top_candidate['Missing Skills'].split(',') if s.strip()]
                if missing_skills_list:
                    missing_categorized_for_viewer = collections.defaultdict(list)
                    for skill in missing_skills_list:
                        found_category = False
                        for category, skills_in_category in SKILL_CATEGORIES.items():
                            if skill.lower() in [s.lower() for s in skills_in_category]:
                                missing_categorized_for_viewer[category].append(skill)
                                found_category = True
                                break
                        if not found_category:
                            missing_categorized_for_viewer["Uncategorized"].append(skill)
                    
                    for category, skills in missing_categorized_for_viewer.items():
                        st.write(f"**{category}:** {', '.join(skills)}")
                else:
                    st.write("No missing skills found for this candidate relative to the JD.")
            else:
                st.write("No missing skills found for this candidate relative to the JD.")

            if top_candidate['Email'] != "Not Found":
                mailto_link_top = create_mailto_link(
                    recipient_email=top_candidate['Email'],
                    candidate_name=top_candidate['Candidate Name'],
                    job_title=top_candidate['JD Used'] if top_candidate['JD Used'] != "Uploaded JD (No file selected)" else "Job Opportunity"
                )
                st.markdown(f'<a href="{mailto_link_top}" target="_blank"><button style="background-color:#00cec9;color:white;border:none;padding:10px 20px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;margin:4px 2px;cursor:pointer;border-radius:8px;">📧 Invite Top Candidate for Interview</button></a>', unsafe_allow_html=True)
            else:
                st.info(f"Email address not found for {top_candidate['Candidate Name']}. Cannot send automated invitation.")
            
            st.markdown("---")
            st.info("For detailed analytics, matched keywords, and missing skills for ALL candidates, please navigate to the **Analytics Dashboard**.")

        else:
            st.info("No candidates processed yet to determine the top candidate.")

        st.markdown("## 🌟 AI Shortlisted Candidates Overview")
        st.caption("Candidates automatically shortlisted based on your score, experience, and CGPA criteria.")

        if not st.session_state['full_results_df'].empty:
            ai_shortlisted_candidates = st.session_state['full_results_df'][
                (st.session_state['full_results_df']['Score (%)'] >= cutoff) & 
                (st.session_state['full_results_df']['Years Experience'] >= min_experience) &
                (st.session_state['full_results_df']['Years Experience'] <= max_experience) &
                ((st.session_state['full_results_df']['CGPA (4.0 Scale)'].isnull()) | (st.session_state['full_results_df']['CGPA (4.0 Scale)'] >= min_cgpa))
            ]

            if not ai_shortlisted_candidates.empty:
                st.success(f"**{len(ai_shortlisted_candidates)}** candidate(s) meet your specified criteria (Score ≥ {cutoff}%, Experience {min_experience}-{max_experience} years, CGPA ≥ {min_cgpa} or N/A).")
                
                display_shortlisted_summary_cols = [
                    'Candidate Name', 'Score (%)', 'Years Experience', 'CGPA (4.0 Scale)',
                    'Semantic Similarity', 'Email', 'AI Suggestion'
                ]
                
                st.dataframe(
                    ai_shortlisted_candidates[display_shortlisted_summary_cols],
                    use_container_width=True, hide_index=True,
                    column_config={
                        "Score (%)": st.column_config.ProgressColumn("Score (%)", format="%f", min_value=0, max_value=100),
                        "Years Experience": st.column_config.NumberColumn("Years Experience", format="%.1f years"),
                        "CGPA (4.0 Scale)": st.column_config.NumberColumn("CGPA (4.0 Scale)", format="%.2f", min_value=0.0, max_value=4.0),
                        "Semantic Similarity": st.column_config.NumberColumn("Semantic Similarity", format="%.2f", min_value=0, max_value=1),
                        "AI Suggestion": st.column_config.Column("AI Suggestion")
                    }
                )
                st.info("For individual detailed AI assessments and action steps, please refer to the table below or the Analytics Dashboard.")
            else:
                st.warning(f"No candidates met the defined screening criteria (score cutoff, experience between {min_experience}-{max_experience} years, and minimum CGPA). You might consider adjusting the sliders or reviewing the uploaded resumes/JD.")
        else:
            st.info("No candidates processed yet for AI shortlisting.")

        st.markdown("---")

        if not st.session_state['full_results_df'].empty:
            st.session_state['full_results_df']['Tag'] = st.session_state['full_results_df'].apply(lambda row: 
                "👑 Exceptional Match" if row['Score (%)'] >= 90 and row['Years Experience'] >= 5 and row['Years Experience'] <= max_experience and row['Semantic Similarity'] >= 0.85 and (row['CGPA (4.0 Scale)'] is None or row['CGPA (4.0 Scale)'] >= 3.5) else (
                "🔥 Strong Candidate" if row['Score (%)'] >= 80 and row['Years Experience'] >= 3 and row['Years Experience'] <= max_experience and row['Semantic Similarity'] >= 0.7 and (row['CGPA (4.0 Scale)'] is None or row['CGPA (4.0 Scale)'] >= 3.0) else (
                "✨ Promising Fit" if row['Score (%)'] >= 60 and row['Years Experience'] >= 1 and row['Years Experience'] <= max_experience and (row['CGPA (4.0 Scale)'] is None or row['CGPA (4.0 Scale)'] >= 2.5) else (
                "⚠️ Needs Review" if row['Score (%)'] >= 40 else 
                "❌ Limited Match"))), axis=1)

        st.markdown("## 📋 Comprehensive Candidate Results Table")
        st.caption("Full details for all processed resumes. Use filters below and edit 'Shortlisted' and 'Notes' directly.")
        
        st.markdown("### Filter Options")
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        filter_col4, filter_col5, filter_col6 = st.columns(3)

        with filter_col1:
            score_min, score_max = st.slider("Score Range (%)", 0, 100, (0, 100), key="score_filter_slider")
        with filter_col2:
            max_exp_for_slider = float(st.session_state['full_results_df']['Years Experience'].max()) if not st.session_state['full_results_df'].empty else 15.0
            exp_min, exp_max = st.slider("Experience Range (Years)", 0.0, max_exp_for_slider, (0.0, max_exp_for_slider), 0.1, key="exp_filter_slider")
        with filter_col3:
            cgpa_min, cgpa_max = st.slider("CGPA Range (4.0 Scale)", 0.0, 4.0, (0.0, 4.0), 0.1, key="cgpa_filter_slider")
        
        with filter_col4:
            all_tags = sorted(st.session_state['full_results_df']['Tag'].unique()) if not st.session_state['full_results_df'].empty else []
            selected_tags = st.multiselect("Filter by Tag", options=all_tags, default=all_tags, key="tag_filter_multiselect")
        
        with filter_col5:
            all_locations = sorted(st.session_state['full_results_df']['Location'].unique()) if not st.session_state['full_results_df'].empty else []
            selected_locations = st.multiselect("Filter by Location", options=all_locations, default=all_locations, key="location_filter_multiselect")
        
        with filter_col6:
            shortlist_filter_options = ["Show All", "Show Shortlisted Only", "Show Non-Shortlisted Only"]
            selected_shortlist_filter = st.selectbox("Filter by Shortlist Status", options=shortlist_filter_options, key="shortlist_filter_selectbox")

        filtered_df_for_editor = st.session_state['full_results_df'].copy()
        
        if not filtered_df_for_editor.empty:
            filtered_df_for_editor = filtered_df_for_editor[
                (filtered_df_for_editor['Score (%)'] >= score_min) & (filtered_df_for_editor['Score (%)'] <= score_max) &
                (filtered_df_for_editor['Years Experience'] >= exp_min) & (filtered_df_for_editor['Years Experience'] <= exp_max) &
                ((filtered_df_for_editor['CGPA (4.0 Scale)'].isnull()) | ((filtered_df_for_editor['CGPA (4.0 Scale)'] >= cgpa_min) & (filtered_df_for_editor['CGPA (4.0 Scale)'] <= cgpa_max)))
            ]
            
            if selected_tags: filtered_df_for_editor = filtered_df_for_editor[filtered_df_for_editor['Tag'].isin(selected_tags)]
            if selected_locations: filtered_df_for_editor = filtered_df_for_editor[filtered_df_for_editor['Location'].apply(lambda x: any(loc in x for loc in selected_locations))]
            if selected_shortlist_filter == "Show Shortlisted Only": filtered_df_for_editor = filtered_df_for_editor[filtered_df_for_editor['Shortlisted'] == True]
            elif selected_shortlist_filter == "Show Non-Shortlisted Only": filtered_df_for_editor = filtered_df_for_editor[filtered_df_for_editor['Shortlisted'] == False]

            jd_raw_skills_set, _ = extract_relevant_keywords(jd_text, jd_skills_for_processing)
            all_unique_jd_skills = sorted(list(jd_raw_skills_set))
            selected_filter_skills = st.multiselect("Filter by Specific Skills (from JD)", options=all_unique_jd_skills, key="skill_filter_multiselect")
            if selected_filter_skills:
                for skill in selected_filter_skills:
                    filtered_df_for_editor = filtered_df_for_editor[filtered_df_for_editor['Matched Keywords'].str.contains(r'\b' + re.escape(skill) + r'\b', case=False, na=False)]

        if st.button("Reset All Filters", key="reset_filters_button"):
            st.session_state['score_filter_slider'] = (0, 100)
            current_max_exp = float(st.session_state['full_results_df']['Years Experience'].max()) if not st.session_state['full_results_df'].empty else 15.0
            st.session_state['exp_filter_slider'] = (0.0, current_max_exp)
            st.session_state['cgpa_filter_slider'] = (0.0, 4.0)
            st.session_state['tag_filter_multiselect'] = sorted(st.session_state['full_results_df']['Tag'].unique()) if not st.session_state['full_results_df'].empty else []
            st.session_state['location_filter_multiselect'] = sorted(st.session_state['full_results_df']['Location'].unique()) if not st.session_state['full_results_df'].empty else []
            st.session_state['shortlist_filter_selectbox'] = "Show All"
            st.session_state['skill_filter_multiselect'] = []
            st.rerun()

        comprehensive_cols = [
            'File Name', # Included for internal use, will be hidden
            'Candidate Name', 'Score (%)', 'Tag', 'AI Suggestion', 'Years Experience',
            'CGPA (4.0 Scale)', 'Semantic Similarity', 'Email', 'Phone Number',
            'Location', 'Languages', 'Matched Keywords', 'Missing Skills',
            'Education Details', 'Work History', 'Project Details', 'JD Used'
        ]
        
        final_display_cols_for_editor = [col for col in comprehensive_cols if col in filtered_df_for_editor.columns]

        if not filtered_df_for_editor.empty:
            edited_df = st.data_editor(
                filtered_df_for_editor[final_display_cols_for_editor + ['Shortlisted', 'Notes']],
                use_container_width=True, hide_index=True,
                column_config={
                    "File Name": st.column_config.Column("File Name", disabled=True, width="hidden"),
                    "Score (%)": st.column_config.ProgressColumn("Score (%)", format="%f", min_value=0, max_value=100),
                    "Years Experience": st.column_config.NumberColumn("Years Experience", format="%.1f years"),
                    "CGPA (4.0 Scale)": st.column_config.NumberColumn("CGPA (4.0 Scale)", format="%.2f", min_value=0.0, max_value=4.0),
                    "Semantic Similarity": st.column_config.NumberColumn("Semantic Similarity", format="%.2f", min_value=0, max_value=1),
                    "AI Suggestion": st.column_config.Column("AI Suggestion", width="medium"),
                    "Tag": st.column_config.Column("Tag", width="small"),
                    "Matched Keywords": st.column_config.Column("Matched Keywords", width="large"),
                    "Missing Skills": st.column_config.Column("Missing Skills", width="large"),
                    "JD Used": st.column_config.Column("JD Used"),
                    "Phone Number": st.column_config.Column("Phone Number"),
                    "Location": st.column_config.Column("Location"),
                    "Languages": st.column_config.Column("Languages"),
                    "Education Details": st.column_config.Column("Education Details", width="large"),
                    "Work History": st.column_config.Column("Work History", width="large"),
                    "Project Details": st.column_config.Column("Project Details", width="large"),
                    "Shortlisted": st.column_config.CheckboxColumn("Shortlist", default=False),
                    "Notes": st.column_config.TextColumn("Notes", default="", max_chars=200)
                },
                key="comprehensive_table_editor"
            )

            if not edited_df.empty:
                edited_subset = edited_df[['File Name', 'Shortlisted', 'Notes']].set_index('File Name')
                current_full_df = st.session_state['full_results_df'].set_index('File Name')
                current_full_df.update(edited_subset)
                st.session_state['full_results_df'] = current_full_df.reset_index()
        else:
            st.info("No candidates match the current filter criteria. Adjust your filters or upload resumes.")

        if not filtered_df_for_editor.empty:
            csv_data = filtered_df_for_editor.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Filtered Results as CSV", data=csv_data,
                file_name="resume_screening_filtered_results.csv", mime="text/csv"
            )
        else:
            st.info("No filtered results to download.")

        st.markdown("---")
        st.markdown("## ⭐ My Manually Shortlisted Candidates")
        st.caption("Candidates you have marked as 'Shortlisted' in the table above.")

        manually_shortlisted_df = st.session_state['full_results_df'][st.session_state['full_results_df']['Shortlisted'] == True]

        if not manually_shortlisted_df.empty:
            final_display_cols_for_shortlisted = [col for col in comprehensive_cols if col in manually_shortlisted_df.columns and col != 'File Name'] # Exclude File Name for display
            st.dataframe(
                manually_shortlisted_df[final_display_cols_for_shortlisted + ['Notes']],
                use_container_width=True, hide_index=True,
                column_config={
                    "Score (%)": st.column_config.ProgressColumn("Score (%)", format="%f", min_value=0, max_value=100),
                    "Years Experience": st.column_config.NumberColumn("Years Experience", format="%.1f years"),
                    "CGPA (4.0 Scale)": st.column_config.NumberColumn("CGPA (4.0 Scale)", format="%.2f", min_value=0.0, max_value=4.0),
                    "Semantic Similarity": st.column_config.NumberColumn("Semantic Similarity", format="%.2f", min_value=0, max_value=1),
                    "AI Suggestion": st.column_config.Column("AI Suggestion", width="medium"),
                    "Tag": st.column_config.Column("Tag", width="small"),
                    "Matched Keywords": st.column_config.Column("Matched Keywords", width="large"),
                    "Missing Skills": st.column_config.Column("Missing Skills", width="large"),
                    "JD Used": st.column_config.Column("JD Used"),
                    "Phone Number": st.column_config.Column("Phone Number"),
                    "Location": st.column_config.Column("Location"),
                    "Languages": st.column_config.Column("Languages"),
                    "Education Details": st.column_config.Column("Education Details", width="large"),
                    "Work History": st.column_config.Column("Work History", width="large"),
                    "Project Details": st.column_config.Column("Project Details", width="large"),
                    "Notes": st.column_config.TextColumn("Notes", width="medium")
                }
            )
            csv_shortlist_data = manually_shortlisted_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Manually Shortlisted Candidates as CSV", data=csv_shortlist_data,
                file_name="manually_shortlisted_candidates.csv", mime="text/csv"
            )
        else:
            st.info("No candidates have been manually shortlisted yet. Use the 'Shortlist' checkbox in the 'Comprehensive Candidate Results Table' above to add them.")

        st.markdown("---")
        st.markdown("## 🔍 Resume Viewer & Detailed Assessment")
        st.caption("Select a candidate to view their raw resume text and full AI assessment.")
        
        candidate_names_for_viewer = ["Select a Candidate"] + sorted(st.session_state['full_results_df']['Candidate Name'].tolist())
        selected_candidate_name_for_viewer = st.selectbox("Choose a candidate to view details:", options=candidate_names_for_viewer, key="resume_viewer_select")

        if selected_candidate_name_for_viewer != "Select a Candidate":
            selected_candidate_row = st.session_state['full_results_df'][st.session_state['full_results_df']['Candidate Name'] == selected_candidate_name_for_viewer].iloc[0]
            
            st.markdown(f"### Raw Resume Text for {selected_candidate_name_for_viewer}")
            st.text_area("Resume Content", selected_candidate_row['Resume Raw Text'], height=400, disabled=True)
            
            st.markdown(f"### Detailed AI Assessment for {selected_candidate_name_for_viewer}")
            st.markdown(selected_candidate_row['Detailed HR Assessment'])

            st.markdown("#### Matched Skills Breakdown:")
            if selected_candidate_row['Matched Keywords (Categorized)']:
                for category, skills in selected_candidate_row['Matched Keywords (Categorized)'].items():
                    st.write(f"**{category}:** {', '.join(skills)}")
            else:
                st.write("No categorized matched skills found.")

            st.markdown("#### Missing Skills Breakdown (from JD):")
            if selected_candidate_row['Missing Skills'].strip():
                missing_skills_list = [s.strip() for s in selected_candidate_row['Missing Skills'].split(',') if s.strip()]
                if missing_skills_list:
                    missing_categorized_for_viewer = collections.defaultdict(list)
                    for skill in missing_skills_list:
                        found_category = False
                        for category, skills_in_category in SKILL_CATEGORIES.items():
                            if skill.lower() in [s.lower() for s in skills_in_category]:
                                missing_categorized_for_viewer[category].append(skill)
                                found_category = True
                                break
                        if not found_category:
                            missing_categorized_for_viewer["Uncategorized"].append(skill)
                    
                    for category, skills in missing_categorized_for_viewer.items():
                        st.write(f"**{category}:** {', '.join(skills)}")
                else:
                    st.write("No missing skills found for this candidate relative to the JD.")
            else:
                st.write("No missing skills found for this candidate relative to the JD.")

            if selected_candidate_row['Email'] != "Not Found":
                mailto_link_selected = create_mailto_link(
                    recipient_email=selected_candidate_row['Email'],
                    candidate_name=selected_candidate_row['Candidate Name'],
                    job_title=selected_candidate_row['JD Used'] if selected_candidate_row['JD Used'] != "Uploaded JD (No file selected)" else "Job Opportunity"
                )
                st.markdown(f'<a href="{mailto_link_selected}" target="_blank"><button style="background-color:#00cec9;color:white;border:none;padding:10px 20px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;margin:4px 2px;cursor:pointer;border-radius:8px;">📧 Invite Selected Candidate for Interview</button></a>', unsafe_allow_html=True)
            else:
                st.info(f"Email address not found for {selected_candidate_name_for_viewer}. Cannot send automated invitation.")

        st.info("Remember to check the Analytics Dashboard for in-depth visualizations of skill overlaps, gaps, and other metrics!")
    else:
        st.info("Please upload a Job Description and at least one Resume to begin the screening process.")

    st.markdown("---")
    if st.button("🗑️ Clear All Processed Data", help="Removes all loaded resumes and screening results from the current session."):
        st.session_state['full_results_df'] = pd.DataFrame(columns=expected_cols)
        st.session_state['full_results_df']['Shortlisted'] = st.session_state['full_results_df']['Shortlisted'].astype(bool)
        st.session_state['full_results_df']['Notes'] = st.session_state['full_results_df']['Notes'].astype(str)
        st.session_state['full_results_df']['Tag'] = st.session_state['full_results_df']['Tag'].astype(str)
        
        for key in ['score_filter_slider', 'exp_filter_slider', 'cgpa_filter_slider',
                    'tag_filter_multiselect', 'location_filter_multiselect',
                    'shortlist_filter_selectbox', 'skill_filter_multiselect',
                    'screening_cutoff_score', 'screening_min_experience',
                    'screening_max_experience', 'screening_min_cgpa',
                    'resume_viewer_select']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# --- Manage JDs Page (Placeholder from manage_jds.py) ---
def manage_jds_page():
    st.markdown('<div class="dashboard-header">📁 Manage Job Descriptions</div>', unsafe_allow_html=True)
    st.write("Here you can add, view, or delete your job description templates.")

    # Ensure 'data' directory exists
    if not os.path.exists("data"):
        os.makedirs("data")

    st.subheader("➕ Add New Job Description")
    with st.form("add_jd_form"):
        jd_name = st.text_input("Job Description Name (e.g., 'Software Engineer')", key="new_jd_name").strip()
        jd_content = st.text_area("Job Description Content", height=300, key="new_jd_content").strip()
        add_jd_button = st.form_submit_button("Add JD")

        if add_jd_button:
            if not jd_name or not jd_content:
                st.error("JD Name and Content cannot be empty.")
            else:
                file_path = os.path.join("data", f"{jd_name.replace(' ', '_').lower()}.txt")
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(jd_content)
                    st.success(f"Job Description '{jd_name}' added successfully!")
                    log_activity(f"User '{st.session_state.username}' added new JD: '{jd_name}'.")
                    st.rerun() # Rerun to update the selectbox
                except Exception as e:
                    st.error(f"Error saving JD: {e}")

    st.markdown("---")
    st.subheader("📋 Existing Job Descriptions")
    jd_files = [f for f in os.listdir("data") if f.endswith(".txt")]
    
    if jd_files:
        jd_options = [f.replace(".txt", "").replace("_", " ").title() for f in jd_files]
        selected_jd_to_view = st.selectbox("Select JD to View/Delete:", ["Select a JD"] + jd_options, key="view_delete_jd_select")

        if selected_jd_to_view != "Select a JD":
            original_file_name = selected_jd_to_view.replace(' ', '_').lower() + ".txt"
            file_path = os.path.join("data", original_file_name)
            
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                st.text_area(f"Content of '{selected_jd_to_view}'", content, height=200, disabled=True)

                if st.button(f"🗑️ Delete '{selected_jd_to_view}'", key="delete_jd_button"):
                    try:
                        os.remove(file_path)
                        st.success(f"Job Description '{selected_jd_to_view}' deleted successfully!")
                        log_activity(f"User '{st.session_state.username}' deleted JD: '{selected_jd_to_view}'.")
                        st.rerun() # Rerun to update the list
                    except Exception as e:
                        st.error(f"Error deleting JD: {e}")
            else:
                st.error(f"File not found for '{selected_jd_to_view}'. It might have been moved or deleted externally.")
    else:
        st.info("No job descriptions found. Add one using the form above.")

# --- Analytics Dashboard Page (from analytics.py) ---
def analytics_dashboard_page():
    st.markdown('<div class="dashboard-header">📊 Screening Analytics</div>', unsafe_allow_html=True)

    if 'full_results_df' not in st.session_state or st.session_state['full_results_df'].empty:
        st.info("No screening results available for analytics. Please run the Resume Screener first.")
        return

    df = st.session_state['full_results_df'].copy()

    st.subheader("Overall Screening Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Resumes Processed", df.shape[0])
    col2.metric("Average Score", f"{df['Score (%)'].mean():.2f}%")
    col3.metric("Average Experience", f"{df['Years Experience'].mean():.1f} years")

    st.markdown("---")

    st.subheader("Candidate Score Distribution")
    fig = px.histogram(df, x="Score (%)", nbins=20, title="Distribution of Candidate Scores",
                       color_discrete_sequence=px.colors.sequential.Teal if not dark_mode else px.colors.sequential.Plasma)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Experience vs. Score")
    fig = px.scatter(df, x="Years Experience", y="Score (%)", color="Tag",
                     title="Experience vs. Score with Candidate Tag", hover_name="Candidate Name",
                     color_discrete_map={
                         "👑 Exceptional Match": "#00cec9",
                         "🔥 Strong Candidate": "#20B2AA",
                         "✨ Promising Fit": "#FFD700",
                         "⚠️ Needs Review": "#FFA500",
                         "❌ Limited Match": "#FF4500"
                     })
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Skill Overlap Analysis")
    if 'Matched Keywords' in df.columns and not df['Matched Keywords'].empty:
        all_matched_skills = []
        for skills_str in df['Matched Keywords'].dropna():
            all_matched_skills.extend([s.strip() for s in skills_str.split(',') if s.strip()])
        
        if all_matched_skills:
            skill_counts = pd.Series(all_matched_skills).value_counts().head(15)
            fig_skills = px.bar(skill_counts, x=skill_counts.values, y=skill_counts.index, orientation='h',
                                title="Top 15 Matched Skills Across All Resumes",
                                labels={'x': 'Frequency', 'y': 'Skill'},
                                color_discrete_sequence=px.colors.sequential.Aggrnyl if not dark_mode else px.colors.sequential.Plasma_r)
            fig_skills.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_skills, use_container_width=True)
        else:
            st.info("No matched skills data available for analysis.")
    else:
        st.info("No 'Matched Keywords' column found or it's empty for skill analysis.")

    st.markdown("---")

    st.subheader("Missing Skills Analysis (Overall)")
    if 'Missing Skills' in df.columns and not df['Missing Skills'].empty:
        all_missing_skills = []
        for skills_str in df['Missing Skills'].dropna():
            all_missing_skills.extend([s.strip() for s in skills_str.split(',') if s.strip()])
        
        if all_missing_skills:
            missing_skill_counts = pd.Series(all_missing_skills).value_counts().head(15)
            fig_missing_skills = px.bar(missing_skill_counts, x=missing_skill_counts.values, y=missing_skill_counts.index, orientation='h',
                                        title="Top 15 Missing Skills Across All Resumes (from JD)",
                                        labels={'x': 'Frequency', 'y': 'Skill'},
                                        color_discrete_sequence=px.colors.sequential.Reds if not dark_mode else px.colors.sequential.Inferno)
            fig_missing_skills.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_missing_skills, use_container_width=True)
        else:
            st.info("No missing skills data available for analysis.")
    else:
        st.info("No 'Missing Skills' column found or it's empty for analysis.")

    st.markdown("---")

    st.subheader("Shortlisted Candidates Breakdown")
    if 'Shortlisted' in df.columns:
        shortlist_counts = df['Shortlisted'].value_counts().reset_index()
        shortlist_counts.columns = ['Shortlisted', 'Count']
        shortlist_counts['Shortlisted'] = shortlist_counts['Shortlisted'].map({True: 'Yes', False: 'No'})
        
        fig_shortlist_pie = px.pie(shortlist_counts, values='Count', names='Shortlisted',
                                   title='Manually Shortlisted Candidates',
                                   color_discrete_sequence=['#00cec9', '#FF6347'] if not dark_mode else ['#20B2AA', '#E9967A'])
        st.plotly_chart(fig_shortlist_pie, use_container_width=True)
    else:
        st.info("No 'Shortlisted' column found in results for analysis.")

# --- Email Candidates Page (Placeholder from email_sender.py) ---
def email_candidates_page():
    st.markdown('<div class="dashboard-header">📤 Email Candidates</div>', unsafe_allow_html=True)
    st.write("Generate pre-filled email templates for shortlisted candidates.")

    if 'full_results_df' not in st.session_state or st.session_state['full_results_df'].empty:
        st.info("No candidates available for emailing. Please run the Resume Screener first.")
        return

    shortlisted_candidates = st.session_state['full_results_df'][st.session_state['full_results_df']['Shortlisted'] == True]

    if shortlisted_candidates.empty:
        st.info("No candidates have been manually shortlisted. Please go to 'Resume Screener' and mark candidates as 'Shortlisted'.")
        return

    st.subheader("Select Candidates to Email")
    
    selected_candidate_names = st.multiselect(
        "Choose candidates:",
        options=shortlisted_candidates['Candidate Name'].tolist(),
        key="email_candidate_select"
    )

    if selected_candidate_names:
        selected_candidates_df = shortlisted_candidates[
            shortlisted_candidates['Candidate Name'].isin(selected_candidate_names)
        ]

        st.subheader("Email Template Options")
        email_subject_template = st.text_input(
            "Email Subject Template:",
            value="Invitation for Interview - {JobTitle} - {CandidateName}",
            help="Use {CandidateName} and {JobTitle} as placeholders."
        )
        email_body_template = st.text_area(
            "Email Body Template:",
            value="""Dear {CandidateName},

We were very impressed with your profile and would like to invite you for an interview for the {JobTitle} position.

Please reply to this email to schedule a suitable time.

Best regards,

The {SenderName}""",
            height=250,
            help="Use {CandidateName}, {JobTitle}, and {SenderName} as placeholders."
        )
        sender_name = st.text_input("Your Name/Company Name (for {SenderName}):", value="Recruiting Team")

        st.markdown("---")
        st.subheader("Generated Email Links")

        for index, row in selected_candidates_df.iterrows():
            candidate_name = row['Candidate Name']
            candidate_email = row['Email']
            job_title_used = row['JD Used'] if row['JD Used'] != "Uploaded JD (No file selected)" else "Job Opportunity"

            if candidate_email == "Not Found":
                st.warning(f"Email address not found for {candidate_name}. Cannot generate link.")
                continue

            # Fill placeholders in subject and body
            subject = email_subject_template.format(CandidateName=candidate_name, JobTitle=job_title_used)
            body = email_body_template.format(CandidateName=candidate_name, JobTitle=job_title_used, SenderName=sender_name)

            mailto_link = create_mailto_link(candidate_email, candidate_name, job_title_used, sender_name)

            st.markdown(f"**{candidate_name}** ({candidate_email})")
            st.markdown(f'<a href="{mailto_link}" target="_blank"><button style="background-color:#00cec9;color:white;border:none;padding:8px 15px;text-align:center;text-decoration:none;display:inline-block;font-size:14px;margin:4px 2px;cursor:pointer;border-radius:8px;">📧 Open Email for {candidate_name}</button></a>', unsafe_allow_html=True)
            st.code(f"Subject: {subject}\n\nBody:\n{body}", language="text")
            st.markdown("---")
        
        if st.button("Log Email Action", key="log_email_action_button"):
            log_activity(f"Generated email links for {len(selected_candidate_names)} candidates.")
            st.success("Email action logged!")

# --- Search Resumes Page (Placeholder from search.py) ---
def search_resumes_page():
    st.markdown('<div class="dashboard-header">🔍 Search Resumes</div>', unsafe_allow_html=True)
    st.write("Search through all processed resumes using keywords.")

    if 'full_results_df' not in st.session_state or st.session_state['full_results_df'].empty:
        st.info("No resumes available to search. Please run the Resume Screener first.")
        return

    df = st.session_state['full_results_df'].copy()

    search_query = st.text_input("Enter keywords to search (e.g., 'Python developer AWS')", key="search_query_input")
    
    if search_query:
        search_results = pd.DataFrame(columns=df.columns)
        query_lower = search_query.lower()

        for index, row in df.iterrows():
            # Search in Candidate Name, Matched Keywords, Missing Skills, and Resume Raw Text
            search_text = f"{row['Candidate Name']} {row['Matched Keywords']} {row['Missing Skills']} {row['Resume Raw Text']}".lower()
            if query_lower in search_text:
                search_results = pd.concat([search_results, pd.DataFrame([row])], ignore_index=True)
        
        if not search_results.empty:
            st.subheader(f"Search Results for '{search_query}'")
            st.dataframe(search_results[[
                'Candidate Name', 'Score (%)', 'Years Experience', 'Matched Keywords', 'Missing Skills', 'AI Suggestion'
            ]], use_container_width=True, hide_index=True)
            log_activity(f"User '{st.session_state.username}' searched for '{search_query}'. Found {len(search_results)} results.")
        else:
            st.info(f"No resumes found matching '{search_query}'.")
    else:
        st.info("Enter keywords in the search bar to find relevant resumes.")

# --- Candidate Notes Page (Placeholder from notes.py) ---
def candidate_notes_page():
    st.markdown('<div class="dashboard-header">📝 Candidate Notes</div>', unsafe_allow_html=True)
    st.write("Add and manage personal notes for each candidate.")

    if 'full_results_df' not in st.session_state or st.session_state['full_results_df'].empty:
        st.info("No candidates available to add notes. Please run the Resume Screener first.")
        return

    df = st.session_state['full_results_df'].copy()

    candidate_names = ["Select a Candidate"] + sorted(df['Candidate Name'].tolist())
    selected_candidate_for_notes = st.selectbox(
        "Select a candidate to add/edit notes:",
        options=candidate_names,
        key="select_candidate_for_notes"
    )

    if selected_candidate_for_notes != "Select a Candidate":
        # Find the current notes for the selected candidate
        current_notes_row = df[df['Candidate Name'] == selected_candidate_for_notes].iloc[0]
        current_notes = current_notes_row['Notes']
        file_name_of_candidate = current_notes_row['File Name']

        st.subheader(f"Notes for {selected_candidate_for_notes}")
        new_notes = st.text_area("Edit Notes:", value=current_notes, height=200, key="notes_text_area")

        if st.button("Save Notes", key="save_notes_button"):
            # Update the notes in the session state DataFrame
            st.session_state['full_results_df'].loc[
                st.session_state['full_results_df']['File Name'] == file_name_of_candidate, 'Notes'
            ] = new_notes
            st.success(f"Notes for {selected_candidate_for_notes} saved successfully!")
            log_activity(f"User '{st.session_state.username}' updated notes for '{selected_candidate_for_notes}'.")
            st.rerun() # Rerun to reflect changes if needed
        
        st.markdown("---")
        st.subheader("All Notes Overview")
        # Display a table of all candidates with their notes
        notes_df = df[['Candidate Name', 'Notes', 'Shortlisted', 'Tag']].copy()
        st.dataframe(notes_df, use_container_width=True, hide_index=True)
    else:
        st.info("Select a candidate from the dropdown to manage their notes.")
