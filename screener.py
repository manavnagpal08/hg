import streamlit as st
import pdfplumber
import re
import os
import sklearn
import joblib
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
# import nltk # Removed NLTK for stopwords
import collections
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse
import uuid
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import tempfile
import shutil
from weasyprint import HTML
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
import traceback
import time
import pandas as pd # Ensure pandas is imported
import requests # For Firestore REST API calls
import json # For JSON parsing/dumping
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS # New, more efficient stop words

# CRITICAL: Disable Hugging Face tokenizers parallelism to avoid deadlocks with ProcessPoolExecutor
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- OCR Specific Imports (Moved to top) ---
from PIL import Image
import pytesseract
import cv2
from pdf2image import convert_from_bytes

# Global NLTK download check (should run once) - No longer needed for stopwords, but kept in case other NLTK components are used later.
# try:
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('stopwords')

# Define global constants
MASTER_CITIES = set([
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

# Changed to use sklearn's ENGLISH_STOP_WORDS for efficiency
STOP_WORDS = set(ENGLISH_STOP_WORDS).union(set([
    "work", "experience", "years", "year", "months", "month", "day", "days", "project", "projects",
    "team", "teams", "developed", "managed", "led", "created", "implemented", "designed",
    "responsible", "proficient", "knowledge", "ability", "strong", "proven", "demonstrated",
    "solution", "solutions", "system", "systems", "platform", "platforms", "framework", "frameworks",
    "database", "databases", "server", "servers", "cloud", "computing", "machine", "learning",
    "artificial", "intelligence", "api", "apis", "rest", "graphql", "agile", "scrum", "kanban",
    "devops", "ci", "cd", "testing", "qa",
    "security", "network", "networking", "virtualization",
    "containerization", "docker", "kubernetes", "terraform", "ansible", "jenkins", "circleci", "github actions", "azure devops", "mlops",
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
    "gcp-pcn", "gcp-psd", "gcp-pda", "gcp-pci", "gcp-pws", "gcp-pwa", "gcp-pme", "gcp-pmc",
    "gcp-pmd", "gcp-pma", "gcp-pmc", "gcp-pmg", "cisco", "juniper", "red", "hat", "rhcsa",
    "rhce", "vmware", "vcpa", "vcpd", "vcpi", "vcpe", "vcpx", "citrix", "cc-v", "cc-p",
    "cc-e", "cc-m", "cc-s", "cc-x", "palo", "alto", "pcnsa", "pcnse", "fortinet", "fcsa",
    "fcsp", "fcc", "fcnsp", "fct", "fcp", "fcs", "fce", "fcn", "fcnp", "fcnse"
]))
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
    ""Docker Compose", "Helm", "Ansible Tower", "SaltStack", "Chef InSpec", "Terraform Cloud", "Vault",
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

# IMPORTANT: REPLACE THESE WITH YOUR ACTUAL DEPLOYMENT URLs
APP_BASE_URL = "https://screenerpro-app.streamlit.app"
CERTIFICATE_HOSTING_URL = "https://manav-jain.github.io/screenerpro-certs"


@st.cache_resource
def get_tesseract_cmd():
    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        return tesseract_path
    return None

# Load ML models once using st.cache_resource
@st.cache_resource
def load_ml_model():
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        ml_model = joblib.load("ml_screening_model.pkl")
        return model, ml_model
    except Exception as e:
        st.error(f"❌ Error loading ML models: {e}. Please ensure 'ml_screening_model.pkl' is in the same directory.")
        return None, None

# Load models globally (once per app run)
global_sentence_model, global_ml_model = load_ml_model()

# Helper for Activity Logging (for screener.py's own activities)
def log_activity_screener(message):
    """Logs an activity with a timestamp to the session state for screener.py's activities."""
    if 'activity_log_screener' not in st.session_state:
        st.session_state.activity_log_screener = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.activity_log_screener.insert(0, f"[{timestamp}] {message}") # Add to the beginning for most recent first
    st.session_state.activity_log_screener = st.session_state.activity_log_screener[:50] # Keep last 50

def preprocess_image_for_ocr(image):
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    img_processed = cv2.adaptiveThreshold(img_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
    return Image.fromarray(img_processed)

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


def extract_text_from_file(file_bytes, file_name, file_type):
    full_text = ""
    # Tesseract configuration for speed and common resume layout
    tesseract_config = "--oem 1 --psm 3" 

    if "pdf" in file_type:
        try:
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                pdf_text = ''.join(page.extract_text() or '' for page in pdf.pages)
            
            if len(pdf_text.strip()) < 50: # Heuristic for potentially scanned PDF
                images = convert_from_bytes(file_bytes)
                for img in images:
                    processed_img = preprocess_image_for_ocr(img)
                    full_text += pytesseract.image_to_string(processed_img, lang='eng', config=tesseract_config) + "\n"
            else:
                full_text = pdf_text

        except Exception as e:
            # Fallback to OCR directly if pdfplumber fails or for any other PDF error
            try:
                images = convert_from_bytes(file_bytes)
                for img in images:
                    processed_img = preprocess_image_for_ocr(img)
                    full_text += pytesseract.image_to_string(processed_img, lang='eng', config=tesseract_config) + "\n"
            except Exception as e_ocr:
                print(f"ERROR: Failed to extract text from PDF via OCR for {file_name}: {str(e_ocr)}")
                return f"[ERROR] Failed to extract text from PDF via OCR: {str(e_ocr)}"

    elif "image" in file_type:
        try:
            img = Image.open(BytesIO(file_bytes)).convert("RGB")
            processed_img = preprocess_image_for_ocr(img)
            full_text = pytesseract.image_to_string(processed_img, lang='eng', config=tesseract_config)
        except Exception as e:
            print(f"ERROR: Failed to extract text from image for {file_name}: {str(e)}")
            return f"[ERROR] Failed to extract text from image: {str(e)}"
    else:
        print(f"ERROR: Unsupported file type for {file_name}: {file_type}")
        return f"[ERROR] Unsupported file type: {file_type}. Please upload a PDF or an image (JPG, PNG)."

    if not full_text.strip():
        print(f"ERROR: No readable text extracted from {file_name}. It might be a very low-quality scan or an empty document.")
        return "[ERROR] No readable text extracted from the file. It might be a very low-quality scan or an empty document."
    
    return full_text


def extract_years_of_experience(text):
    text = text.lower()
    total_months = 0
    
    date_patterns = [
        r'(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})\s*(?:to|–|-)\s*(present|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})',
        r'(\b\d{4})\s*(?:to|–|-)\s*(present|\b\d{4})'
    ]

    for pattern in date_patterns:
        job_date_ranges = re.findall(pattern, text)
        for start_str, end_str in job_date_ranges:
            start_date = None
            end_date = None

            try:
                start_date = datetime.strptime(start_str.strip(), '%B %Y')
            except ValueError:
                try:
                    start_date = datetime.strptime(start_str.strip(), '%b %Y')
                except ValueError:
                    try:
                        start_date = datetime(int(start_str.strip()), 1, 1)
                    except ValueError:
                        pass

            if start_date is None:
                continue

            if end_str.strip() == 'present':
                end_date = datetime.now()
            else:
                try:
                    end_date = datetime.strptime(end_str.strip(), '%B %Y')
                except ValueError:
                    try:
                        end_date = datetime.strptime(end_str.strip(), '%b %Y')
                    except ValueError:
                        try:
                            end_date = datetime(int(end_str.strip()), 12, 31)
                        except ValueError:
                            pass
            
            if end_date is None:
                continue

            delta_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
            total_months += max(delta_months, 0)

    if total_months > 0:
        return round(total_months / 12, 1)
    else:
        match = re.search(r'(\d+(?:\.\d+)?)\s*(\+)?\s*(year|yrs|years)\b', text)
        if not match:
            match = re.search(r'experience[^\d]{0,10}(\d+(?:\.\d+)?)', text)
        if match:
            return float(match.group(1))

    return 0.0

def extract_email(text):
    text = text.lower()

    text = text.replace("gmaill.com", "gmail.com").replace("gmai.com", "gmail.com")
    text = text.replace("yah00", "yahoo").replace("outiook", "outlook")
    text = text.replace("coim", "com").replace("hotmai", "hotmail")

    text = re.sub(r'[^\w\s@._+-]', ' ', text)

    possible_emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.\w+', text)

    if possible_emails:
        for email in possible_emails:
            if "gmail" in email or "manav" in email: # Specific filter, consider removing or making configurable
                return email
        return possible_emails[0]
    
    return None

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

    if found_locations:
        return ", ".join(sorted(list(found_locations)))
    return "Not Found"

def extract_name(text):
    lines = text.strip().split('\n')
    if not lines:
        return None

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
        if name:
            return name.title()
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
        else:
            continue

        if scale and scale not in [0, 1]:
            normalized_cgpa = (raw_cgpa / scale) * 4.0
            return round(normalized_cgpa, 2)
        elif raw_cgpa <= 4.0:
            return round(raw_cgpa, 2)
        elif raw_cgpa <= 10.0:
            return round((raw_cgpa / 10.0) * 4.0, 2)
        
    return None

def extract_education_text(text):
    """
    Extracts a single-line education entry from resume text.
    Returns a clean string like: "B.Tech in CSE, Alliance University, Bangalore – 2028"
    Works with or without 'Expected' in the year.
    """

    text = text.replace('\r', '').replace('\t', ' ')
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]

    education_section = ''
    capture = False

    for line in lines:
        line_lower = line.lower()
        if any(h in line_lower for h in ['education', 'academic background', 'qualifications']):
            capture = True
            continue
        if capture and any(h in line_lower for h in ['experience', 'skills', 'certifications', 'projects', 'languages']):
            break
        if capture:
            education_section += line + ' '

    education_section = education_section.strip()

    edu_match = re.search(
        r'([A-Za-z0-9.,()&\-\s]+?(university|college|institute|school)[^–\n]{0,50}[–\-—]?\s*(expected\s*)?\d{4})',
        education_section,
        re.IGNORECASE
    )

    if edu_match:
        return edu_match.group(1).strip()

    fallback_match = re.search(
        r'([A-Za-z0-9.,()&\-\s]+?(b\.tech|m\.tech|b\.sc|m\.sc|bca|bba|mba|ph\.d)[^–\n]{0,50}\d{4})',
        education_section,
        re.IGNORECASE
    )
    if fallback_match:
        return fallback_match.group(1).strip()

    fallback_line = education_section.split('.')[0].strip()
    return fallback_line if fallback_line else None

def extract_work_history(text):
    work_history_section_matches = re.finditer(r'(?:experience|work history|employment history)\s*(\n|$)', text, re.IGNORECASE)
    work_details = []
    
    start_index = -1
    for match in work_history_section_matches:
        start_index = match.end()
        break

    if start_index != -1:
        sections = ['education', 'skills', 'projects', 'certifications', 'awards', 'publications']
        end_index = len(text)
        for section in sections:
            section_match = re.search(r'\b' + re.escape(section) + r'\b', text[start_index:], re.IGNORECASE)
            if section_match:
                end_index = start_index + section_match.start()
                break
        
        work_text = text[start_index:end_index].strip()
        
        job_blocks = re.split(r'\n(?=[A-Z][a-zA-Z\s,&\.]+(?:\s(?:at|@))?\s*[A-Z][a-zA-Z\s,&\.]*\s*(?:-|\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}))', work_text, flags=re.IGNORECASE)
        
        for block in job_blocks:
            block = block.strip()
            if not block:
                continue
            
            company = None
            title = None
            start_date = None
            end_date = None

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
                if title_company_match:
                    title = title_company_match.group(1).strip()
                    company = title_company_match.group(2).strip()
                    break
                
                company_title_match = re.search(r'^([A-Z][a-zA-Z\s,\-&.]+),\s*([A-Z][a-zA-Z\s,\-&.]+)', line)
                if company_title_match:
                    company = company_title_match.group(1).strip()
                    title = company_title_match.group(2).strip()
                    break
                
                if not company and not title:
                    potential_org_match = re.search(r'^[A-Z][a-zA-Z\s,\-&.]+', line)
                    if potential_org_match and len(potential_org_match.group(0).split()) > 1:
                        if not company: company = potential_org_match.group(0).strip()
                        elif not title: title = potential_org_match.group(0).strip()
                        break

            if company or title or start_date or end_date:
                work_details.append({
                    "Company": company,
                    "Title": title,
                    "Start Date": start_date,
                    "End Date": end_date
                })
    return work_details

def extract_project_details(text, MASTER_SKILLS):
    """
    Extracts real project entries from resume text.
    Returns a list of dicts: Title, Description, Technologies Used
    """

    project_details = []

    text = text.replace('\r', '').replace('\t', ' ')
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]

    # Step 1: Isolate project section
    project_section_keywords = r'(projects|personal projects|key projects|portfolio|selected projects|major projects|academic projects|relevant projects)'
    project_section_match = re.search(project_section_keywords + r'\s*(\n|$)', text, re.IGNORECASE)

    if not project_section_match:
        project_text = text[:1000]  # fallback to first 1000 chars
        start_index = 0
    else:
        start_index = project_section_match.end()
        sections = ['education', 'experience', 'skills', 'certifications', 'awards', 'publications', 'interests', 'hobbies']
        end_index = len(text)
        for section in sections:
            match = re.search(r'\b' + re.escape(section) + r'\b', text[start_index:], re.IGNORECASE)
            if match:
                end_index = start_index + match.start()
                break
        project_text = text[start_index:end_index].strip()

    if not project_text:
        return []

    lines = [line.strip() for line in project_text.split('\n') if line.strip()]
    current_project = {"Project Title": None, "Description": [], "Technologies Used": set()}

    forbidden_title_keywords = [
        'skills gained', 'responsibilities', 'reflection', 'summary',
        'achievements', 'capabilities', 'what i learned', 'tools used'
    ]

    for i, line in enumerate(lines):
        line_lower = line.lower()

        # Skip all-uppercase names or headers
        if re.match(r'^[A-Z\s]{5,}$', line) and len(line.split()) <= 4:
            continue

        # Previous line was a bullet?
        prev_line_is_bullet = False
        if i > 0 and re.match(r'^[•*-]', lines[i - 1]):
            prev_line_is_bullet = True

        # Strong new project title if:
        # - starts with number or bullet
        # - not just a soft skill block
        # - contains 3–15 words
        # - not all caps
        is_title = (
            (re.match(r'^[•*-]?\s*\d+[\).:-]?\s', line) or line.lower().startswith("project")) and
            3 <= len(line.split()) <= 15 and
            not any(kw in line_lower for kw in forbidden_title_keywords) and
            not prev_line_is_bullet and
            not line.isupper()
        )

        is_url = re.match(r'https?://', line_lower)

        # New Project Begins
        if is_title or is_url:
            if current_project["Project Title"] or current_project["Description"]:
                full_desc = "\n".join(current_project["Description"])
                techs, _ = extract_relevant_keywords(full_desc, MASTER_SKILLS)
                current_project["Technologies Used"].update(techs)

                project_details.append({
                    "Project Title": current_project["Project Title"],
                    "Description": full_desc.strip(),
                    "Technologies Used": ", ".join(sorted(current_project["Technologies Used"]))
                })

            current_project = {"Project Title": line, "Description": [], "Technologies Used": set()}
        else:
            current_project["Description"].append(line)

    # Add last project
    if current_project["Project Title"] or current_project["Description"]:
        full_desc = "\n".join(current_project["Description"])
        techs, _ = extract_relevant_keywords(full_desc, MASTER_SKILLS)
        current_project["Technologies Used"].update(techs)

        project_details.append({
            "Project Title": current_project["Project Title"],
            "Description": full_desc.strip(),
            "Technologies Used": ", ".join(sorted(current_project["Technologies Used"]))
        })

    return project_details


def extract_languages(text):
    """
    Extracts known languages from resume text.
    Returns a comma-separated string of detected languages or 'Not Found'.
    """
    languages_list = set()
    cleaned_full_text = clean_text(text)

    # De-duplicated, lowercase language set
    all_languages = list(set([
        "english", "hindi", "spanish", "french", "german", "mandarin", "japanese", "arabic",
        "russian", "portuguese", "italian", "korean", "bengali", "marathi", "telugu", "tamil",
        "gujarati", "urdu", "kannada", "odia", "malayalam", "punjabi", "assamese", "kashmiri",
        "sindhi", "sanskrit", "dutch", "swedish", "norwegian", "danish", "finnish", "greek",
        "turkish", "hebrew", "thai", "vietnamese", "indonesian", "malay", "filipino", "swahili",
        "farsi", "persian", "polish", "ukrainian", "romanian", "czech", "slovak", "hungarian",
        "chinese", "tagalog", "nepali", "sinhala", "burmese", "khmer", "lao", "pashto", "dari",
        "uzbek", "kazakh", "azerbaijani", "georgian", "armenian", "albanian", "serbian",
        "croatian", "bosnian", "bulgarian", "macedonian", "slovenian", "estonian", "latvian",
        "lithuanian", "icelandic", "irish", "welsh", "gaelic", "maltese", "esperanto", "latin",
        "ancient greek", "modern greek", "yiddish", "romani", "catalan", "galician", "basque",
        "breton", "cornish", "manx", "frisian", "luxembourgish", "sami", "romansh", "sardinian",
        "corsican", "occitan", "provencal", "walloon", "flemish", "afrikaans", "zulu", "xhosa",
        "sesotho", "setswana", "shona", "ndebele", "venda", "tsonga", "swati", "kikuyu",
        "luganda", "kinyarwanda", "kirundi", "lingala", "kongo", "yoruba", "igbo", "hausa",
        "fulani", "twi", "ewe", "ga", "dagbani", "gur", "mossi", "bambara", "senufo", "wolof",
        "mandinka", "susu", "krio", "temne", "limba", "mende", "gola", "vai", "kpelle", "loma",
        "bandi", "bassa", "grebo", "krahn", "dan", "mano", "guerze", "kono", "kisi"
    ]))

    sorted_all_languages = sorted(all_languages, key=len, reverse=True)

    # Step 1: Try to locate a language-specific section
    section_match = re.search(
        r'\b(languages|language skills|linguistic abilities|known languages)\s*[:\-]?\s*\n?',
        cleaned_full_text, re.IGNORECASE
    )

    if section_match:
        start_index = section_match.end()
        # Optional: stop at next known section
        end_index = len(cleaned_full_text)
        stop_words = ['education', 'experience', 'skills', 'certifications', 'awards', 'publications', 'interests', 'hobbies']
        for stop in stop_words:
            m = re.search(r'\b' + stop + r'\b', cleaned_full_text[start_index:], re.IGNORECASE)
            if m:
                end_index = start_index + m.start()
                break

        language_chunk = cleaned_full_text[start_index:end_index]
    else:
        language_chunk = cleaned_full_text

    # Step 2: Match known languages
    for lang in sorted_all_languages:
        # Use word boundaries for exact matches and allow for common suffixes like " (fluent)"
        pattern = r'\b' + re.escape(lang) + r'(?:\s*\(?[a-z\s,-]+\)?)?\b'
        if re.search(pattern, language_chunk, re.IGNORECASE):
            if lang == "de":
                languages_list.add("German")
            else:
                languages_list.add(lang.title())

    return ", ".join(sorted(languages_list)) if languages_list else "Not Found"


def format_work_history(work_list):
    if not work_list:
        return "Not Found"
    formatted_entries = []
    for entry in work_list:
        parts = []
        if entry.get("Title"):
            parts.append(entry["Title"])
        if entry.get("Company"):
            parts.append(f"at {entry['Company']}")
        if entry.get("Start Date") and entry.get("End Date"):
            parts.append(f"({entry['Start Date']} - {entry['End Date']})")
        elif entry.get("Start Date"):
            parts.append(f"(Since {entry['Start Date']})")
        formatted_entries.append(" ".join(parts).strip())
    return "; ".join(formatted_entries) if formatted_entries else "Not Found"

def format_project_details(proj_list):
    if not proj_list:
        return "Not Found"
    formatted_entries = []
    for entry in proj_list:
        parts = []
        if entry.get("Project Title"):
            parts.append(f"**{entry['Project Title']}**")
        if entry.get("Technologies Used"):
            parts.append(f"({entry['Technologies Used']})")
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

    high_score = 85
    moderate_score = 65
    high_exp = 4
    moderate_exp = 2
    high_sem_sim = 0.75
    moderate_sem_sim = 0.4
    high_cgpa = 3.5
    moderate_cgpa = 3.0

    if score >= high_score and years_exp >= high_exp and semantic_similarity >= high_sem_sim:
        overall_fit_description = "High alignment."
        key_strength_hint = "Strong technical and experience match, quick integration expected."
        review_focus_text = "Cultural fit, project contributions."
    elif score >= moderate_score and years_exp >= moderate_exp and semantic_similarity >= moderate_sem_sim:
        overall_fit_description = "Good potential."
        key_strength_hint = "Solid foundation, good growth prospects."
        review_focus_text = "Specific skill gaps, long-term career goals."
    else:
        overall_fit_description = "Consider further review."
        key_strength_hint = "May require development in key areas."
        review_focus_text = "Fundamental skills, foundational knowledge."

    if cgpa is not None:
        if cgpa >= high_cgpa:
            key_strength_hint += " Excellent academic record."
        elif cgpa >= moderate_cgpa:
            key_strength_hint += " Strong academic background."

    suggestion_text = (
        f"**Candidate: {candidate_name}**\n\n"
        f"**Overall Fit:** {overall_fit_description}\n"
        f"**Key Strengths:** {key_strength_hint}\n"
        f"**Focus for Review:** {review_focus_text}"
    )
    return suggestion_text

def save_certificate_to_firestore_rest(certificate_data, firestore_rest_api_base_url, firebase_web_api_key, app_id):
    """
    Saves certificate data to Firestore using the REST API.
    Data is stored in a public collection: /artifacts/{appId}/public/data/certificates/{certificate_id}
    """
    certificate_id = certificate_data.get("certificate_id")
    if not certificate_id:
        log_activity_screener("❌ Error: Certificate ID missing for Firestore save.")
        return False

    collection_path = f"artifacts/{app_id}/public/data/certificates"
    document_path = f"{collection_path}/{certificate_id}"
    
    # Firestore REST API URL for a specific document
    url = f"{firestore_rest_api_base_url}/projects/{app_id}/databases/(default)/documents/{document_path}?key={firebase_web_api_key}"

    headers = {
        "Content-Type": "application/json"
    }
    
    # The data to send to Firestore
    # Convert sets to lists for JSON serialization
    data_to_save = {k: list(v) if isinstance(v, set) else v for k, v in certificate_data.items()}

    try:
        response = requests.patch(url, headers=headers, data=json.dumps({"fields": data_to_save}))
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        log_activity_screener(f"✅ Certificate {certificate_id} saved to Firestore successfully.")
        return True
    except requests.exceptions.RequestException as e:
        log_activity_screener(f"❌ Error saving certificate {certificate_id} to Firestore: {e}")
        log_activity_screener(f"Response content: {response.text if 'response' in locals() else 'N/A'}")
        return False


def generate_certificate_html(candidate_name, score, rank, current_date, certificate_id):
    certificate_date = current_date.strftime("%B %d, %Y")
    
    # Encode certificate_id for URL
    encoded_cert_id = urllib.parse.quote_plus(certificate_id)
    # Construct verification URL
    verification_url = f"{CERTIFICATE_HOSTING_URL}/?id={encoded_cert_id}"

    # Generate QR code URL (using Google Charts API for simplicity, can be replaced with a local library if needed)
    qr_code_url = f"https://chart.googleapis.com/chart?chs=150x150&cht=qr&chl={urllib.parse.quote_plus(verification_url)}"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Certificate of Achievement</title>
        <link href="https://fonts.googleapis.com/css2?family=Merriweather:wght@700&family=Open+Sans:wght@400;600&display=swap" rel="stylesheet">
        <style>
            body {{
                font-family: 'Open Sans', sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f0f2f5;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
            }}
            .certificate-container {{
                width: 297mm; /* A4 width */
                height: 210mm; /* A4 height */
                padding: 20mm;
                box-sizing: border-box;
                background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
                border: 15px solid #0056b3;
                border-image: linear-gradient(45deg, #0056b3, #007bff, #0056b3) 1;
                box-shadow: 0 0 20px rgba(0, 0, 0, 0.2);
                text-align: center;
                position: relative;
                overflow: hidden;
            }}
            .certificate-container::before {{
                content: '';
                position: absolute;
                top: -50px;
                left: -50px;
                right: -50px;
                bottom: -50px;
                background: url('https://www.transparenttextures.com/patterns/cubes.png') repeat; /* Subtle background pattern */
                opacity: 0.05;
                z-index: 0;
            }}
            .content {{
                position: relative;
                z-index: 1;
            }}
            h1 {{
                font-family: 'Merriweather', serif;
                font-size: 3.5em;
                color: #0056b3;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            }}
            h2 {{
                font-family: 'Merriweather', serif;
                font-size: 2.5em;
                color: #333;
                margin-top: 5px;
                margin-bottom: 20px;
            }}
            p {{
                font-size: 1.2em;
                color: #555;
                line-height: 1.6;
                margin-bottom: 15px;
            }}
            .name {{
                font-family: 'Merriweather', serif;
                font-size: 3em;
                color: #007bff;
                margin: 20px 0;
                padding-bottom: 5px;
                border-bottom: 3px dashed #bbb;
                display: inline-block;
                text-transform: capitalize;
            }}
            .score-rank {{
                font-size: 1.8em;
                color: #0056b3;
                margin-top: 20px;
                font-weight: 600;
            }}
            .date {{
                font-size: 1.1em;
                color: #777;
                margin-top: 30px;
            }}
            .signature-section {{
                display: flex;
                justify-content: space-around;
                align-items: flex-end;
                margin-top: 50px;
            }}
            .signature-box {{
                text-align: center;
                width: 45%;
            }}
            .signature-line {{
                border-top: 1px solid #aaa;
                margin-top: 40px;
                margin-bottom: 5px;
            }}
            .signature-text {{
                font-size: 0.9em;
                color: #666;
            }}
            .footer-note {{
                font-size: 0.8em;
                color: #999;
                margin-top: 40px;
            }}
            .qr-code {{
                position: absolute;
                bottom: 20mm;
                right: 20mm;
                text-align: center;
            }}
            .qr-code img {{
                border: 2px solid #0056b3;
            }}
            .qr-code p {{
                font-size: 0.7em;
                color: #555;
                margin-top: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="certificate-container">
            <div class="content">
                <h1>CERTIFICATE OF ACHIEVEMENT</h1>
                <p>This certifies that</p>
                <div class="name">{candidate_name}</div>
                <p>has successfully demonstrated outstanding performance in the</p>
                <h2>ScreenerPro Assessment</h2>
                <p>achieving an impressive score of</p>
                <div class="score-rank">Score: {score}% | Rank: {rank}</div>
                <p>This accomplishment reflects a high level of skill and dedication.</p>
                <div class="date">Awarded on {certificate_date}</div>
                <div class="signature-section">
                    <div class="signature-box">
                        <div class="signature-line"></div>
                        <div class="signature-text">ScreenerPro Team Lead</div>
                    </div>
                    <div class="signature-box">
                        <div class="signature-line"></div>
                        <div class="signature-text">Head of Talent Acquisition</div>
                    </div>
                </div>
                <p class="footer-note">Certificate ID: {certificate_id}</p>
            </div>
            <div class="qr-code">
                <img src="{qr_code_url}" alt="QR Code" width="100" height="100">
                <p>Scan to Verify</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

def send_email_with_certificate(recipient_email, candidate_name, certificate_pdf_bytes):
    sender_email = os.environ.get("EMAIL_SENDER_ADDRESS")
    sender_password = os.environ.get("EMAIL_SENDER_PASSWORD")

    if not sender_email or not sender_password:
        log_activity_screener("❌ Email sender credentials not configured. Skipping email.")
        st.warning("Email sender credentials not configured. Certificate email will not be sent.")
        return False

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = f"ScreenerPro Certificate for {candidate_name}"

    body = f"""
    Dear {candidate_name},

    Congratulations on successfully completing the ScreenerPro assessment!

    Please find your Certificate of Achievement attached.

    You can also verify your certificate online by scanning the QR code on the certificate or by visiting the verification page.

    Best regards,
    The ScreenerPro Team
    """
    msg.attach(MIMEText(body, 'plain'))

    part = MIMEBase('application', 'octet-stream')
    part.set_payload(certificate_pdf_bytes)
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f"attachment; filename=ScreenerPro_Certificate_{candidate_name.replace(' ', '_')}.pdf")
    msg.attach(part)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        log_activity_screener(f"✅ Certificate email sent to {recipient_email}")
        return True
    except Exception as e:
        log_activity_screener(f"❌ Failed to send email to {recipient_email}: {e}")
        st.error(f"Failed to send email to {recipient_email}: {e}")
        return False

# Function to run in parallel for text extraction
def _parallel_extract_text(file_data):
    file_bytes, file_name, file_type = file_data
    try:
        extracted_text = extract_text_from_file(file_bytes, file_name, file_type)
        return file_name, extracted_text, None # Return None for error if successful
    except Exception as e:
        return file_name, None, str(e) # Return error message if extraction fails

# Modified resume_screener_page function signature
def resume_screener_page(firestore_rest_api_base_url, firebase_web_api_key, app_id):
    st.title("📄 Resume Screener Pro")
    st.markdown("Upload resumes (PDF or Image) and a job description to find the best candidates.")

    if 'resume_data' not in st.session_state:
        st.session_state.resume_data = []
    if 'job_desc_text' not in st.session_state:
        st.session_state.job_desc_text = ""
    if 'job_desc_embedding' not in st.session_state:
        st.session_state.job_desc_embedding = None

    # Job Description Input
    with st.expander("Job Description", expanded=True):
        st.markdown("**Enter the Job Description:**")
        new_job_desc_text = st.text_area("Job Description", height=200, label_visibility="collapsed", key="job_desc_input")
        
        if new_job_desc_text and new_job_desc_text != st.session_state.job_desc_text:
            st.session_state.job_desc_text = new_job_desc_text
            cleaned_jd = clean_text(new_job_desc_text)
            if global_sentence_model:
                st.session_state.job_desc_embedding = global_sentence_model.encode([cleaned_jd])[0]
                st.success("Job Description processed and embedded!")
            else:
                st.error("SentenceTransformer model not loaded. Cannot process Job Description.")
                st.session_state.job_desc_embedding = None
        elif not new_job_desc_text:
            st.session_state.job_desc_text = ""
            st.session_state.job_desc_embedding = None
            st.info("Please enter a job description to begin screening.")

    # Resume Upload
    st.markdown("---")
    st.markdown("**Upload Resumes (PDF or Image files):**")
    uploaded_files = st.file_uploader("Choose files", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True, label_visibility="collapsed")

    if uploaded_files and st.session_state.job_desc_embedding is not None:
        if st.button("🚀 Start Screening"):
            st.session_state.resume_data = [] # Clear previous results
            
            # Removed progress_bar and status_text updates from within the loop for speed
            # progress_bar = st.progress(0, text="Starting resume processing...")
            # status_text = st.empty()
            
            st.info(f"Starting processing for {len(uploaded_files)} resumes. This may take a moment...")

            all_file_data_for_extraction = []
            for i, uploaded_file in enumerate(uploaded_files):
                file_bytes = uploaded_file.read()
                file_name = uploaded_file.name
                file_type = uploaded_file.type
                all_file_data_for_extraction.append((file_bytes, file_name, file_type))

            extracted_results = []
            extracted_texts_for_embedding = []
            resume_metadata = [] # To store original file info and extracted text

            start_time = time.time()

            # Phase 1: Parallel Text Extraction
            # status_text.info(f"Phase 1/2: Extracting text from {total_files} resumes in parallel...") # Removed
            log_activity_screener(f"Starting parallel text extraction for {len(uploaded_files)} resumes.")
            
            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(_parallel_extract_text, data): data[1] for data in all_file_data_for_extraction}
                
                for i, future in enumerate(as_completed(futures)):
                    file_name_original = futures[future]
                    try:
                        file_name, extracted_text, error = future.result()
                        if error:
                            st.error(f"Error processing {file_name}: {error}")
                            log_activity_screener(f"Error extracting text from {file_name}: {error}")
                        else:
                            extracted_results.append((file_name, extracted_text))
                            extracted_texts_for_embedding.append(clean_text(extracted_text))
                            log_activity_screener(f"Text extracted from {file_name}.")
                    except Exception as e:
                        st.error(f"Unexpected error during text extraction for {file_name_original}: {e}")
                        log_activity_screener(f"Unexpected error during text extraction for {file_name_original}: {e}")
                    
                    # progress_bar.progress((i + 1) / total_files, text=f"Extracted text from {i+1}/{total_files} resumes...") # Removed

            extraction_end_time = time.time()
            log_activity_screener(f"Text extraction phase completed in {extraction_end_time - start_time:.2f} seconds.")

            # Phase 2: Batch Embedding and Further Processing
            if extracted_texts_for_embedding:
                # status_text.info(f"Phase 2/2: Generating embeddings and analyzing {len(extracted_texts_for_embedding)} resumes...") # Removed
                log_activity_screener(f"Starting batch embedding for {len(extracted_texts_for_embedding)} resumes.")
                
                # Generate all embeddings in one batch
                all_resume_embeddings = global_sentence_model.encode(extracted_texts_for_embedding, show_progress_bar=False) # Changed to False
                
                log_activity_screener("Batch embedding completed.")

                # Process each resume sequentially after getting its embedding
                for i, (file_name, full_text) in enumerate(extracted_results):
                    if "[ERROR]" in full_text:
                        st.error(f"Skipping {file_name} due to prior text extraction error.")
                        st.session_state.resume_data.append({
                            "File Name": file_name,
                            "Status": "Error: Text extraction failed",
                            "Score": 0,
                            "Semantic Similarity": 0,
                            "AI Suggestion": "N/A",
                            "Extracted Skills": "N/A",
                            "Categorized Skills": {},
                            "Years of Experience": 0,
                            "Email": "N/A",
                            "Phone": "N/A",
                            "Location": "N/A",
                            "Name": "N/A",
                            "CGPA (4.0 Scale)": "N/A",
                            "Education": "N/A",
                            "Work History": "N/A",
                            "Projects": "N/A",
                            "Languages": "N/A"
                        })
                        continue

                    try:
                        cleaned_text = clean_text(full_text)
                        resume_embedding = all_resume_embeddings[i] # Get the pre-calculated embedding

                        # Calculate semantic similarity
                        semantic_similarity = cosine_similarity([resume_embedding], [st.session_state.job_desc_embedding])[0][0]

                        # Prepare features for ML model
                        years_exp = extract_years_of_experience(full_text)
                        
                        # Use the trained ML model to predict a score
                        # Ensure features are in the correct order and format (e.g., 2D array)
                        features = np.array([[semantic_similarity, years_exp]])
                        if global_ml_model:
                            predicted_score_proba = global_ml_model.predict_proba(features)[0][1] * 100
                            score = round(predicted_score_proba, 2)
                        else:
                            score = round(semantic_similarity * 100, 2) # Fallback if ML model not loaded

                        extracted_skills, categorized_skills = extract_relevant_keywords(full_text, MASTER_SKILLS)
                        email = extract_email(full_text)
                        phone = extract_phone_number(full_text)
                        location = extract_location(full_text)
                        name = extract_name(full_text)
                        cgpa = extract_cgpa(full_text)
                        education = extract_education_text(full_text)
                        work_history = extract_work_history(full_text)
                        projects = extract_project_details(full_text, MASTER_SKILLS)
                        languages = extract_languages(full_text)

                        ai_suggestion = generate_concise_ai_suggestion(name if name else file_name, score, years_exp, semantic_similarity, cgpa)

                        st.session_state.resume_data.append({
                            "File Name": file_name,
                            "Status": "Processed",
                            "Score": score,
                            "Semantic Similarity": round(semantic_similarity, 3),
                            "AI Suggestion": ai_suggestion,
                            "Extracted Skills": ", ".join(sorted(extracted_skills)),
                            "Categorized Skills": categorized_skills,
                            "Years of Experience": years_exp,
                            "Email": email,
                            "Phone": phone,
                            "Location": location,
                            "Name": name,
                            "CGPA (4.0 Scale)": cgpa,
                            "Education": education,
                            "Work History": format_work_history(work_history),
                            "Projects": format_project_details(projects),
                            "Languages": languages,
                            "Full Text": full_text # Store full text for detailed view
                        })

                        # Certificate Generation and Saving
                        if score >= 75: # Example threshold for certificate
                            certificate_id = str(uuid.uuid4())
                            certificate_data = {
                                "certificate_id": certificate_id,
                                "candidate_name": name if name else file_name,
                                "score": score,
                                "rank": "Top Performer", # This would ideally come from ranking all candidates
                                "date_issued": datetime.now().strftime("%Y-%m-%d"),
                                "job_description_hash": hash(st.session_state.job_desc_text), # Hash of JD for uniqueness
                                "semantic_similarity": semantic_similarity,
                                "years_experience": years_exp,
                                "skills": list(extracted_skills),
                                "email": email,
                                "education": education,
                                "work_history_summary": format_work_history(work_history),
                                "projects_summary": format_project_details(projects),
                                "location": location
                            }
                            
                            # Save to Firestore via REST API
                            save_certificate_to_firestore_rest(certificate_data, firestore_rest_api_base_url, firebase_web_api_key, app_id)

                            # Generate and optionally send PDF certificate
                            certificate_html = generate_certificate_html(name if name else file_name, score, "Top Performer", date.today(), certificate_id)
                            
                            # Convert HTML to PDF
                            pdf_bytes = BytesIO()
                            HTML(string=certificate_html).write_pdf(pdf_bytes)
                            pdf_bytes.seek(0)
                            
                            st.download_button(
                                label=f"Download Certificate for {name if name else file_name}",
                                data=pdf_bytes.getvalue(),
                                file_name=f"ScreenerPro_Certificate_{name.replace(' ', '_')}.pdf",
                                mime="application/pdf",
                                key=f"download_cert_{file_name}"
                            )
                            
                            if email and st.checkbox(f"Send certificate to {email} for {name if name else file_name}?", key=f"send_email_{file_name}"):
                                send_email_with_certificate(email, name if name else file_name, pdf_bytes.getvalue())
                                st.success(f"Email sent to {email} for {name if name else file_name}!")

                    except Exception as e:
                        st.error(f"Error processing {file_name}: {e}")
                        traceback.print_exc()
                        log_activity_screener(f"Error processing {file_name}: {e}")
                        st.session_state.resume_data.append({
                            "File Name": file_name,
                            "Status": f"Error: {e}",
                            "Score": 0,
                            "Semantic Similarity": 0,
                            "AI Suggestion": "N/A",
                            "Extracted Skills": "N/A",
                            "Categorized Skills": {},
                            "Years of Experience": 0,
                            "Email": "N/A",
                            "Phone": "N/A",
                            "Location": "N/A",
                            "Name": "N/A",
                            "CGPA (4.0 Scale)": "N/A",
                            "Education": "N/A",
                            "Work History": "N/A",
                            "Projects": "N/A",
                            "Languages": "N/A"
                        })
                    
                    # progress_bar.progress((i + 1) / len(extracted_results), text=f"Analyzing {i+1}/{len(extracted_results)} resumes...") # Removed
            else:
                st.warning("No valid text extracted from uploaded resumes for embedding.")
            
            end_time = time.time()
            total_processing_time = end_time - start_time
            st.success(f"Processing complete! Total time: {total_processing_time:.2f} seconds for {len(st.session_state.resume_data)} resumes.") # Final success message
            log_activity_screener(f"Total processing time: {total_processing_time:.2f} seconds for {len(st.session_state.resume_data)} resumes.")
            # progress_bar.empty() # Removed
            # status_text.empty() # Removed

    if st.session_state.resume_data:
        st.subheader("Screening Results")
        
        # Convert to DataFrame for better display and filtering
        df = pd.DataFrame(st.session_state.resume_data)
        
        # Filter out error rows for main display, but keep them in raw data
        display_df = df[df['Status'] == 'Processed'].copy()
        
        if not display_df.empty:
            # Sort by score by default
            display_df = display_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
            display_df["Rank"] = display_df.index + 1 # Add rank based on sorted order

            st.dataframe(
                display_df[[
                    "Rank", "File Name", "Name", "Score", "Semantic Similarity", 
                    "Years of Experience", "Location", "Email", "Phone", "AI Suggestion"
                ]].style.format({
                    "Score": "{:.2f}%",
                    "Semantic Similarity": "{:.3f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Detailed Candidate Insights")
            selected_candidate_name = st.selectbox("Select a candidate for detailed view:", display_df["Name"].tolist() if not display_df["Name"].isnull().all() else display_df["File Name"].tolist())

            if selected_candidate_name:
                # Find the selected candidate's full data
                selected_candidate_data = display_df[(display_df["Name"] == selected_candidate_name) | (display_df["File Name"] == selected_candidate_name)].iloc[0]

                st.markdown(f"### Details for {selected_candidate_data['Name'] if selected_candidate_data['Name'] else selected_candidate_data['File Name']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Overall Score", f"{selected_candidate_data['Score']:.2f}%")
                    st.metric("Semantic Similarity", f"{selected_candidate_data['Semantic Similarity']:.3f}")
                    st.metric("Years of Experience", f"{selected_candidate_data['Years of Experience']} years")
                    st.metric("CGPA (4.0 Scale)", selected_candidate_data['CGPA (4.0 Scale)'] if selected_candidate_data['CGPA (4.0 Scale)'] else "N/A")
                    st.metric("Location", selected_candidate_data['Location'])
                with col2:
                    st.metric("Email", selected_candidate_data['Email'] if selected_candidate_data['Email'] else "N/A")
                    st.metric("Phone", selected_candidate_data['Phone'] if selected_candidate_data['Phone'] else "N/A")
                    st.metric("Education", selected_candidate_data['Education'] if selected_candidate_data['Education'] else "N/A")
                    st.metric("Languages", selected_candidate_data['Languages'])
                
                st.markdown("---")
                st.subheader("AI Suggestion")
                st.markdown(selected_candidate_data['AI Suggestion'])

                st.subheader("Work History")
                st.markdown(selected_candidate_data['Work History'] if selected_candidate_data['Work History'] else "Not Found")

                st.subheader("Projects")
                st.markdown(selected_candidate_data['Projects'] if selected_candidate_data['Projects'] else "Not Found")

                st.subheader("Extracted Skills")
                st.write(selected_candidate_data['Extracted Skills'] if selected_candidate_data['Extracted Skills'] else "No specific skills extracted.")
                
                st.subheader("Categorized Skills Breakdown")
                categorized_skills = selected_candidate_data['Categorized Skills']
                if categorized_skills:
                    for category, skills_list in categorized_skills.items():
                        if skills_list:
                            st.markdown(f"**{category}:** {', '.join(sorted(set(skills_list)))}")
                else:
                    st.write("No categorized skills found.")

                st.subheader("Raw Resume Text")
                with st.expander("View Full Extracted Text"):
                    st.text(selected_candidate_data['Full Text'])

                # Skill Cloud Visualization
                st.subheader("Skill Cloud")
                extracted_skills_text = selected_candidate_data['Extracted Skills']
                if extracted_skills_text and extracted_skills_text != "N/A":
                    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(extracted_skills_text)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig) # Close the plot to prevent memory issues
                else:
                    st.info("Upload resumes and screen to see the skill cloud.")
        else:
            st.info("No resumes were successfully processed. Please check for errors above.")

    st.markdown("---")
    st.markdown("## 📊 Candidate Score Comparison")
    st.caption("Visual overview of how each candidate ranks against the job requirements.")
    dark_mode = st.session_state.get("dark_mode_main", False)

    if not st.session_state['comprehensive_df'].empty:
        # New condition: Only display graph if number of resumes is less than 25
        if len(st.session_state['comprehensive_df']) < 25:
            fig, ax = plt.subplots(figsize=(12, 7))
            colors = ['#4CAF50' if s >= cutoff else '#FFC107' if s >= (cutoff * 0.75) else '#F44346' for s in st.session_state['comprehensive_df']['Score (%)']]
            bars = ax.bar(st.session_state['comprehensive_df']['Candidate Name'], st.session_state['comprehensive_df']['Score (%)'], color=colors)
            ax.set_xlabel("Candidate", fontsize=14, color='white' if dark_mode else 'black')
            ax.set_ylabel("Score (%)", fontsize=14, color='white' if dark_mode else 'black')
            ax.set_title("Resume Screening Scores Across Candidates", fontsize=16, fontweight='bold', color='white' if dark_mode else 'black')
            ax.set_ylim(0, 100)
            plt.xticks(rotation=60, ha='right', fontsize=10, color='white' if dark_mode else 'black')
            plt.yticks(fontsize=10, color='white' if dark_mode else 'black')
            ax.tick_params(axis='x', colors='white' if dark_mode else 'black')
            ax.tick_params(axis='y', colors='white' if dark_mode else 'black')

            if dark_mode:
                fig.patch.set_facecolor('#1E1E1E')
                ax.set_facecolor('#2D2D2D')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['right'].set_color('white')
            
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}", ha='center', va='bottom', fontsize=9, color='white' if dark_mode else 'black')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info(f"Graph not displayed for {len(st.session_state['comprehensive_df'])} resumes. Displaying graphs for fewer than 25 resumes provides a clearer visualization.")
    else:
        st.info("Upload resumes to see a comparison chart.")

    st.markdown("---")

    st.markdown("## 👑 Top Candidate AI Assessment")
    st.caption("A concise, AI-powered assessment for the most suitable candidate.")
    
    if not st.session_state['comprehensive_df'].empty:
        top_candidate = st.session_state['comprehensive_df'].iloc[0]
        
        cgpa_display = f"{top_candidate['CGPA (4.0 Scale)']:.2f}" if pd.notna(top_candidate['CGPA (4.0 Scale)']) else "N/A"
        semantic_sim_display = f"{top_candidate['Semantic Similarity']:.2f}" if pd.notna(top_candidate['Semantic Similarity']) else "N/A"

        st.markdown(f"### **{top_candidate['Candidate Name']}**")
        st.markdown(f"**Score:** {top_candidate['Score (%)']:.2f}% | **Experience:** {top_candidate['Years Experience']:.1f} years | **CGPA:** {cgpa_display} (4.0 Scale) | **Semantic Similarity:** {semantic_sim_display}")
        
        if top_candidate['Certificate Rank'] != "Not Applicable":
            st.markdown(f"**ScreenerPro Certification:** {top_candidate['Certificate Rank']}")

        st.markdown(f"**AI Assessment:**")
        st.markdown(top_candidate['Detailed HR Assessment'])
        
        st.markdown("#### Matched Skills Breakdown:")
        if top_candidate['Matched Keywords (Categorized)']:
            if isinstance(top_candidate['Matched Keywords (Categorized)'], dict):
                for category, skills in top_candidate['Matched Keywords (Categorized)'].items():
                    st.write(f"**{category}:** {', '.join(skills)}")
            else:
                st.write(f"Raw Matched Keywords: {top_candidate['Matched Keywords']}")
        else:
            st.write("No categorized matched skills found.")

        st.markdown("#### Missing Skills Breakdown (from JD):")
        jd_raw_skills_set, jd_categorized_skills_for_top = extract_relevant_keywords(jd_text, all_master_skills)
        resume_raw_skills_set_for_top, _ = extract_relevant_keywords(top_candidate['Resume Raw Text'], all_master_skills)
        
        missing_skills_for_top = jd_raw_skills_set.difference(resume_raw_skills_set_for_top)
        
        if missing_skills_for_top:
            missing_categorized = collections.defaultdict(list)
            for skill in missing_skills_for_top:
                found_category = False
                for category, skills_in_category in SKILL_CATEGORIES.items():
                    if skill.lower() in [s.lower() for s in skills_in_category]:
                        missing_categorized[category].append(skill)
                        found_category = True
                        break
                if not found_category:
                    missing_categorized["Uncategorized"].append(skill)
            
            if missing_categorized:
                for category, skills in missing_categorized.items():
                    st.write(f"**{category}:** {', '.join(skills)}")
            else:
                st.write("No categorized missing skills found for this candidate relative to the JD.")
        else:
            st.write("No missing skills found for this candidate relative to the JD.")


        if top_candidate['Email'] != "Not Found":
            mailto_link_top = create_mailto_link(
                recipient_email=top_candidate['Email'],
                candidate_name=top_candidate['Candidate Name'],
                job_title=jd_name_for_results if jd_name_for_results != "Uploaded JD (No file selected)" else "Job Opportunity"
            )
            st.markdown(f'<a href="{mailto_link_top}" target="_blank"><button style="background-color:#00cec9;color:white;border:none;padding:10px 20px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;margin:4px 2px;cursor:pointer;border-radius:8px;">📧 Invite Top Candidate for Interview</button></a>', unsafe_allow_html=True)
        else:
            st.info(f"Email address not found for {top_candidate['Candidate Name']}. Cannot send automated invitation.")
        
        st.markdown("---")
        st.info("For detailed analytics, matched keywords, and missing skills for ALL candidates, please navigate to the **Analytics Dashboard**.")

    else:
        st.info("No candidates processed yet to determine the top candidate.")


    st.markdown("## 🌟 Candidates Meeting Criteria Overview")
    st.caption("Candidates automatically identified as meeting your defined score, experience, and CGPA criteria.")

    auto_shortlisted_candidates = st.session_state['comprehensive_df'][
        (st.session_state['comprehensive_df']['Score (%)'] >= cutoff) & 
        (st.session_state['comprehensive_df']['Years Experience'] >= min_experience) &
        (st.session_state['comprehensive_df']['Years Experience'] <= max_experience)
    ].copy()

    if 'CGPA (4.0 Scale)' in auto_shortlisted_candidates.columns and auto_shortlisted_candidates['CGPA (4.0 Scale)'].notnull().any():
        auto_shortlisted_candidates = auto_shortlisted_candidates[
            (auto_shortlisted_candidates['CGPA (4.0 Scale)'].isnull()) | (auto_shortlisted_candidates['CGPA (4.0 Scale)'] >= min_cgpa)
        ]

    if not auto_shortlisted_candidates.empty:
        st.success(f"**{len(auto_shortlisted_candidates)}** candidate(s) meet your specified criteria (Score ≥ {cutoff}%, Experience {min_experience}-{max_experience} years, and minimum CGPA ≥ {min_cgpa} or N/A).")
        
        display_auto_shortlisted_cols = [
            'Candidate Name',
            'Score (%)',
            'Years Experience',
            'CGPA (4.0 Scale)',
            'Semantic Similarity',
            'Email',
            'AI Suggestion',
            'Certificate Rank'
        ]
        
        st.dataframe(
            auto_shortlisted_candidates[display_auto_shortlisted_cols],
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
                "Semantic Similarity": st.column_config.NumberColumn(
                    "Semantic Similarity",
                    help="Conceptual similarity between JD and Resume (higher is better)",
                    format="%.2f",
                    min_value=0,
                    max_value=1
                ),
                "AI Suggestion": st.column_config.Column(
                    "AI Suggestion",
                    help="AI's concise overall assessment and recommendation"
                ),
                "Certificate Rank": st.column_config.Column(
                    "Certificate Rank",
                    help="ScreenerPro Certification Level"
                )
            }
        )
        st.info("For individual detailed AI assessments and action steps, please refer to the table below.")

    else:
        st.warning(f"No candidates met the defined screening criteria (score cutoff, experience between {min_experience}-{max_experience} years, and minimum CGPA). You might consider adjusting the sliders or reviewing the uploaded resumes/JD.")

    st.markdown("---")

    st.markdown("## 📋 Comprehensive Candidate Results Table")
    st.caption("Full details for all processed resumes. Use the filters below to refine the view.")
    
    st.markdown("### 🔍 Filter Candidates")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    filter_col4, filter_col5, filter_col6 = st.columns(3)

    with filter_col1:
        jd_raw_skills_set, _ = extract_relevant_keywords(jd_text, all_master_skills)
        all_unique_jd_skills = sorted(list(jd_raw_skills_set))
        selected_filter_skills = st.multiselect(
            "**Skills (AND logic):**",
            options=all_unique_jd_skills,
            help="Only candidates possessing ALL selected skills will be shown."
        )
    with filter_col2:
        search_query = st.text_input(
            "**Keyword Search:**",
            placeholder="Name, Email, Location, Raw Text...",
            help="Search for text across Candidate Name, Email, Location, and Resume Raw Text."
        )
    with filter_col3:
        selected_tags = st.multiselect(
            "**AI Tag:**",
            options=["👑 Exceptional Match", "🔥 Strong Candidate", "✨ Promising Fit", "⚠️ Needs Review", "❌ Limited Match"],
            help="Filter by AI-generated assessment tags."
        )
    
    with filter_col4:
        min_score_filter, max_score_filter = st.slider(
            "**Score Range (%):**",
            0, 100, (0, 100), key="score_range_filter", help="Filter candidates by their overall score range."
        )
    with filter_col5:
        min_exp_filter, max_exp_filter = st.slider(
            "**Experience Range (Years):**",
            0, 20, (0, 20), key="exp_range_filter", help="Filter candidates by their years of experience range."
        )
    with filter_col6:
        min_cgpa_filter, max_cgpa_filter = st.slider(
            "**CGPA Range (4.0 Scale):**",
            0.0, 4.0, (0.0, 4.0), 0.1, key="cgpa_range_filter", help="Filter candidates by their CGPA range (normalized to 4.0)."
        )
    
    filter_col_loc, filter_col_lang = st.columns(2)
    with filter_col_loc:
        all_locations = sorted(st.session_state['comprehensive_df']['Location'].unique())
        selected_locations = st.multiselect(
            "**Location:**",
            options=all_locations,
            help="Filter by candidate location."
        )
    with filter_col_lang:
        all_languages_from_df = sorted(list(set(
            lang.strip() for langs_str in st.session_state['comprehensive_df']['Languages'] if langs_str != "Not Found" for lang in langs_str.split(',')
        )))
        selected_languages = st.multiselect(
            "**Languages:**",
            options=all_languages_from_df,
            help="Filter by languages spoken by the candidate."
        )


    filtered_display_df = st.session_state['comprehensive_df'].copy()

    if selected_filter_skills:
        for skill in selected_filter_skills:
            filtered_display_df = filtered_display_df[filtered_display_df['Matched Keywords'].str.contains(r'\b' + re.escape(skill) + r'\b', case=False, na=False)]

    if search_query:
        search_query_lower = search_query.lower()
        filtered_display_df = filtered_display_df[
            filtered_display_df['Candidate Name'].str.lower().str.contains(search_query_lower, na=False) |
            filtered_display_df['Email'].str.lower().str.contains(search_query_lower, na=False) |
            filtered_display_df['Phone Number'].str.lower().str.contains(search_query_lower, na=False) |
            filtered_display_df['Location'].str.lower().str.contains(search_query_lower, na=False) |
            filtered_display_df['Resume Raw Text'].str.lower().str.contains(search_query_lower, na=False)
        ]
    
    if selected_tags:
        filtered_display_df = filtered_display_df[filtered_display_df['Tag'].isin(selected_tags)]
    
    filtered_display_df = filtered_display_df[
        (filtered_display_df['Score (%)'] >= min_score_filter) & (filtered_display_df['Score (%)'] <= max_score_filter)
    ]
    filtered_display_df = filtered_display_df[
        (filtered_display_df['Years Experience'] >= min_exp_filter) & (filtered_display_df['Years Experience'] <= max_exp_filter)
    ]
    if not filtered_display_df.empty and 'CGPA (4.0 Scale)' in filtered_display_df.columns:
        if not (min_cgpa_filter == 0.0 and max_cgpa_filter == 4.0):
            filtered_display_df = filtered_display_df[
                ((filtered_display_df['CGPA (4.0 Scale)'].notnull()) & 
                 (filtered_display_df['CGPA (4.0 Scale)'] >= min_cgpa_filter) & 
                 (filtered_display_df['CGPA (4.0 Scale)'] <= max_cgpa_filter))
            ]
    
    if selected_locations:
        location_pattern = '|'.join([re.escape(loc) for loc in selected_locations])
        filtered_display_df = filtered_display_df[
            filtered_display_df['Location'].str.contains(location_pattern, case=False, na=False)
        ]
    
    if selected_languages:
        language_pattern = '|'.join([re.escape(lang) for lang in selected_languages])
        filtered_display_df = filtered_display_df[
            filtered_display_df['Languages'].str.contains(language_pattern, case=False, na=False)
        ]

    comprehensive_cols = [
        'Candidate Name',
        'Score (%)',
        'Years Experience',
        'CGPA (4.0 Scale)',
        'Email',
        'Phone Number',
        'Location',
        'Languages',
        'Education Details',
        'Work History',
        'Project Details',
        'Semantic Similarity',
        'Tag',
        'AI Suggestion',
        'Certificate Rank',
        'Matched Keywords',
        'Missing Skills',
        'JD Used',
        'Date Screened',
        'Certificate ID'
    ]
    
    final_display_cols = [col for col in comprehensive_cols if col in filtered_display_df.columns]

    st.dataframe(
        filtered_display_df[final_display_cols],
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
            "Semantic Similarity": st.column_config.NumberColumn(
                "Semantic Similarity",
                help="Conceptual similarity between JD and Resume (higher is better)",
                format="%.2f",
                min_value=0,
                max_value=1
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
            "Matched Keywords": st.column_config.Column(
                "Matched Keywords",
                help="Keywords found in both JD and Resume"
            ),
            "Missing Skills": st.column_config.Column(
                "Missing Skills",
                help="Key skills from JD not found in Resume"
            ),
            "JD Used": st.column_config.Column(
                "JD Used",
                help="Job Description used for this screening"
            ),
            "Date Screened": st.column_config.DateColumn(
                "Date Screened",
                help="Date when the resume was screened",
                format="YYYY-MM-DD"
            ),
            "Phone Number": st.column_config.Column(
                "Phone Number",
                help="Candidate's phone number extracted from resume"
            ),
            "Location": st.column_config.Column(
                "Location",
                help="Candidate's location extracted from resume"
            ),
            "Languages": st.column_config.Column(
                "Languages",
                help="Languages spoken by the candidate"
            ),
            "Education Details": st.column_config.Column(
                "Education Details",
                help="Structured education history (University, Degree, Major, Year)"
            ),
            "Work History": st.column_config.Column(
                "Work History",
                help="Structured work experience (Company, Title, Dates)"
            ),
            "Project Details": st.column_config.Column(
                "Project Details",
                help="Structured project experience (Title, Description, Technologies)"
            ),
            "Certificate ID": st.column_config.Column(
                "Certificate ID",
                help="Unique ID for the certificate",
                disabled=True,
                width="hidden"
            )
        }
    )
    
    st.markdown("---")
    st.markdown("## 🏆 Generate Candidate Certificates")
    st.caption("Select a candidate to view or download their ScreenerPro Certification.")

    if not st.session_state['comprehensive_df'].empty:
        candidate_names_for_cert = st.session_state['comprehensive_df']['Candidate Name'].tolist()
        selected_candidate_name_for_cert = st.selectbox(
            "**Select Candidate for Certificate:**",
            options=candidate_names_for_cert,
            key="certificate_candidate_select"
        )

        if selected_candidate_name_for_cert:
            candidate_rows = st.session_state['comprehensive_df'][
                st.session_state['comprehensive_df']['Candidate Name'] == selected_candidate_name_for_cert
            ]
            
            if not candidate_rows.empty:
                candidate_data_for_cert = candidate_rows.iloc[0].to_dict()

                if candidate_data_for_cert.get('Certificate Rank') != "Not Applicable":
                    # Save certificate data to Firestore
                    save_certificate_to_firestore_rest(candidate_data_for_cert, firestore_rest_api_base_url, firebase_web_api_key, app_id)

                    # Generate HTML content for the certificate
                    certificate_html_content = generate_certificate_html(candidate_data_for_cert['Candidate Name'], candidate_data_for_cert['Score (%)'], candidate_data_for_cert['Certificate Rank'], date.today(), candidate_data_for_cert['Certificate ID'])
                    st.session_state['certificate_html_content'] = certificate_html_content # Store for preview

                    col_cert_view, col_cert_download, col_cert_email_option = st.columns(3)
                    with col_cert_view:
                        if st.button("👁️ View Certificate", key="view_cert_button"):
                            # This button just triggers the preview, content is already generated
                            pass 
                            
                    with col_cert_download:
                        pdf_bytes = BytesIO()
                        HTML(string=certificate_html_content).write_pdf(pdf_bytes)
                        pdf_bytes.seek(0)
                        st.download_button(
                            label="⬇️ Download Certificate (PDF)",
                            data=pdf_bytes.getvalue(),
                            file_name=f"ScreenerPro_Certificate_{candidate_data_for_cert['Candidate Name'].replace(' ', '_')}.pdf",
                            mime="application/pdf",
                            key="download_cert_button"
                        )
                    
                    with col_cert_email_option:
                        attach_html = st.checkbox("Attach HTML to Email?", key="attach_html_checkbox", help="If checked, the full HTML certificate file will be attached to the email. Note: Dynamic features (QR code, live verification) may not work when opened locally from attachment.")

                    # Automatically send email if certificate is generated and email is available
                    if candidate_data_for_cert.get('Email') and candidate_data_for_cert['Email'] != "Not Found":
                        # For send_email_with_certificate, we need the PDF bytes, not HTML content directly
                        pdf_bytes_for_email = BytesIO()
                        HTML(string=certificate_html_content).write_pdf(pdf_bytes_for_email)
                        pdf_bytes_for_email.seek(0)

                        send_email_with_certificate(
                            recipient_email=candidate_data_for_cert['Email'],
                            candidate_name=candidate_data_for_cert['Candidate Name'],
                            certificate_pdf_bytes=pdf_bytes_for_email.getvalue()
                        )
                    else:
                        st.info(f"No email address found for {candidate_data_for_cert['Candidate Name']}. Certificate could not be sent automatically.")

                else:
                    st.info(f"{selected_candidate_name_for_cert} does not qualify for a ScreenerPro Certificate at this time.")
            else:
                st.warning(f"Selected candidate '{selected_candidate_name_for_cert}' not found in the processed results. Please re-select or re-process resumes.")
    else:
        st.info("No candidates available to generate certificates for. Please screen resumes first.")

    if st.session_state['certificate_html_content']:
        st.markdown("---")
        st.markdown("### Generated Certificate Preview")
        st.components.v1.html(st.session_state['certificate_html_content'], height=600, scrolling=True)
        st.markdown("---")


    else:
        st.info("Please upload a Job Description and at least one Resume to begin the screening process.")

