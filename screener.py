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
import nltk
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

# CRITICAL: Disable Hugging Face tokenizers parallelism to avoid deadlocks with ProcessPoolExecutor
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- OCR Specific Imports (Moved to top) ---
from PIL import Image
import pytesseract
import cv2
from pdf2image import convert_from_bytes

# Global NLTK download check (should run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

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
        overall_fit_description = "Moderate fit."
        key_strength_hint = "Good foundational skills, potential for growth."
        review_focus_text = "Depth of experience, skill application, learning agility."
    else:
        overall_fit_description = "Limited alignment."
        key_strength_hint = "May require significant development or a different role."
        review_focus_text = "Foundational skills, transferable experience, long-term potential."

    cgpa_note = ""
    if cgpa is not None:
        if cgpa >= high_cgpa:
            cgpa_note = "Excellent academic record. "
        elif cgpa >= moderate_cgpa:
            cgpa_note = "Solid academic background. "
        else:
            cgpa_note = "Academic record may need review. "
    else:
        cgpa_note = "CGPA not found. "

    summary_text = f"**Fit:** {overall_fit_description} **Strengths:** {cgpa_note}{key_strength_hint} **Focus:** {review_focus_text}"
    return summary_text

@st.cache_data(show_spinner="Generating detailed HR Assessment...")
def generate_detailed_hr_assessment(candidate_name, score, years_exp, semantic_similarity, cgpa, jd_text, resume_text, matched_keywords, missing_skills, max_exp_cutoff):
    assessment_parts = []
    overall_assessment_title = ""
    next_steps_focus = ""

    matched_kws_str = ", ".join(matched_keywords) if isinstance(matched_keywords, list) else matched_keywords
    missing_skills_str = ", ".join(missing_skills) if isinstance(missing_skills, list) else missing_skills

    high_score = 90
    strong_score = 80
    promising_score = 60
    high_exp = 5
    strong_exp = 3
    promising_exp = 1
    high_sem_sim = 0.85
    strong_sem_sim = 0.7
    promising_sem_sim = 0.35
    high_cgpa = 3.5
    strong_cgpa = 3.0
    promising_cgpa = 2.5

    if score >= high_score and years_exp >= high_exp and years_exp <= max_exp_cutoff and semantic_similarity >= high_sem_sim and (cgpa is None or cgpa >= high_cgpa):
        overall_assessment_title = "Exceptional Candidate: Highly Aligned with Strategic Needs"
        assessment_parts.append(f"**{candidate_name}** presents an **exceptional profile** with a high score of {score:.2f}% and {years_exp:.1f} years of experience. This demonstrates a profound alignment with the job description's core requirements, further evidenced by a strong semantic similarity of {semantic_similarity:.2f}.")
        if cgpa is not None:
            assessment_parts.append(f"Their academic record, with a CGPA of {cgpa:.2f} (normalized to 4.0 scale), further solidifies their strong foundational knowledge.")
        assessment_parts.append(f"**Key Strengths:** This candidate possesses a robust skill set directly matching critical keywords in the JD, including: *{matched_kws_str if matched_kws_str else 'No specific keywords listed, but overall strong match'}*. Their extensive experience indicates a capacity for leadership and handling complex challenges, suggesting immediate productivity and minimal ramp-up time. They are poised to make significant contributions from day one.")
        assessment_parts.append("The resume highlights a clear career progression and a history of successful project delivery, often exceeding expectations. Their qualifications exceed expectations, making them a top-tier applicant for this role.")
        assessment_parts.append("This individual's profile suggests they are not only capable of fulfilling the role's duties but also have the potential to mentor others, drive innovation, and take on strategic initiatives within the team. Their background indicates a strong fit for a high-impact position.")
        next_steps_focus = "The next steps should focus on assessing cultural integration, exploring leadership potential, and delving into strategic contributions during the interview. Prepare for a deep dive into their most challenging projects, how they navigated complex scenarios, and their long-term vision. Consider fast-tracking this candidate through the interview process and potentially involving senior leadership early on."
        assessment_parts.append(f"**Action:** Strongly recommend for immediate interview. Prioritize for hiring and consider for advanced roles if applicable.")

    elif score >= strong_score and years_exp >= strong_exp and years_exp <= max_exp_cutoff and semantic_similarity >= strong_sem_sim and (cgpa is None or cgpa >= strong_cgpa):
        overall_assessment_title = "Strong Candidate: Excellent Potential for Key Contributions"
        assessment_parts.append(f"**{candidate_name}** is a **strong candidate** with a score of {score:.2f}% and {years_exp:.1f} years of experience. They show excellent alignment with the job description, supported by a solid semantic similarity of {semantic_similarity:.2f}.")
        if cgpa is not None:
            assessment_parts.append(f"Their academic performance, with a CGPA of {cgpa:.2f}, indicates a solid theoretical grounding.")
        assessment_parts.append(f"**Key Strengths:** Significant overlap in required skills and practical experience that directly addresses the job's demands. Matched keywords include: *{matched_kws_str if matched_kws_str else 'No specific keywords listed, but overall strong match'}*. This individual is likely to integrate well and contribute effectively from an early stage, bringing valuable expertise to the team.")
        assessment_parts.append("Their resume indicates a consistent track record of achieving results and adapting to new challenges. They demonstrate a solid understanding of the domain and could quickly become a valuable asset, requiring moderate onboarding.")
        assessment_parts.append("This candidate is well-suited for the role and demonstrates the core competencies required. Their experience suggests they can handle typical challenges and contribute positively to team dynamics.")
        next_steps_focus = "During the interview, explore specific project methodologies, problem-solving approaches, and long-term career aspirations to confirm alignment with team dynamics and growth opportunities within the company. Focus on behavioral questions to understand their collaboration style, initiative, and how they handle feedback. A technical assessment might be beneficial to confirm depth of skills."
        assessment_parts.append(f"**Action:** Recommend for interview. Good fit for the role, with potential for growth.")

    elif score >= promising_score and years_exp >= promising_exp and years_exp <= max_exp_cutoff and semantic_similarity >= promising_sem_sim and (cgpa is None or cgpa >= promising_cgpa):
        overall_assessment_title = "Promising Candidate: Requires Focused Review on Specific Gaps"
        assessment_parts.append(f"**{candidate_name}** is a **promising candidate** with a score of {score:.2f}% and {years_exp:.1f} years of experience. While demonstrating a foundational understanding (semantic similarity: {semantic_similarity:.2f}), there are areas that warrant deeper investigation to ensure a complete fit.")
        
        gaps_identified = []
        if score < 70:
            gaps_identified.append("The overall score suggests some core skill areas may need development or further clarification.")
        if years_exp < promising_exp:
            gaps_identified.append(f"Experience ({years_exp:.1f} yrs) is on the lower side; assess their ability to scale up quickly and take on more responsibility.")
        if semantic_similarity < 0.5:
            gaps_identified.append("Semantic understanding of the JD's nuances might be limited; probe their theoretical knowledge versus practical application in real-world scenarios.")
        if cgpa is not None and cgpa < promising_cgpa:
            gaps_identified.append(f"Academic record (CGPA: {cgpa:.2f}) is below preferred, consider its relevance to role demands.")
        if missing_skills_str:
            gaps_identified.append(f"**Potential Missing Skills:** *{missing_skills_str}*. Focus interview questions on these areas to assess their current proficiency or learning agility.")
        
        if years_exp > max_exp_cutoff:
            gaps_identified.append(f"Experience ({years_exp:.1f} yrs) exceeds the maximum desired ({max_exp_cutoff} yrs). Evaluate if this indicates overqualification or a potential mismatch in role expectations.")

        if gaps_identified:
            assessment_parts.append("Areas for further exploration include: " + " ".join(gaps_identified))
        
        assessment_parts.append("The candidate shows potential, especially if they can demonstrate quick learning or relevant transferable skills. Their resume indicates a willingness to grow and take on new challenges, which is a positive sign for development opportunities.")
        next_steps_focus = "The interview should focus on validating foundational skills, understanding their learning agility, and assessing their potential for growth within the role. Be prepared to discuss specific examples of how they've applied relevant skills and how they handle challenges, particularly in areas where skills are missing. Consider a skills assessment or a structured case study to gauge problem-solving abilities. Discuss their motivation for this role and long-term career goals."
        assessment_parts.append(f"**Action:** Consider for initial phone screen or junior role. Requires careful evaluation and potentially a development plan.")

    else:
        overall_assessment_title = "Limited Match: Consider Only for Niche Needs or Pipeline Building"
        assessment_parts.append(f"**{candidate_name}** shows a **limited match** with a score = {score:.2f}% and {years_exp:.1f} years of experience (semantic similarity: {semantic_similarity:.2f}). This profile indicates a significant deviation from the core requirements of the job description.")
        if cgpa is not None:
            assessment_parts.append(f"Their academic record (CGPA: {cgpa:.2f}) also indicates a potential mismatch.")
        assessment_parts.append(f"**Key Concerns:** A low overlap in essential skills and potentially insufficient experience for the role's demands. Many key skills appear to be missing: *{missing_skills_str if missing_skills_str else 'No specific missing skills listed, but overall low match'}*. While some transferable skills may exist, a substantial investment in training or a re-evaluation of role fit would likely be required for this candidate to succeed.")
        
        if years_exp > max_exp_cutoff:
            assessment_parts.append(f"Additionally, their experience ({years_exp:.1f} yrs) significantly exceeds the maximum desired ({max_exp_cutoff} yrs), which might indicate overqualification or a mismatch in career trajectory for this specific opening.")

        assessment_parts.append("The resume does not strongly align with the technical or experience demands of this specific position. Their background may be more suited for a different type of role or industry, or an entry-level position if their core skills are strong but experience is lacking.")
        assessment_parts.append("This candidate might not be able to meet immediate role requirements without extensive support. Their current profile suggests a mismatch with the current opening.")
        next_steps_focus = "This candidate is generally not recommended for the current role unless there are specific, unforeseen niche requirements or a strategic need to broaden the candidate pool significantly. If proceeding, focus on understanding their fundamental capabilities, their motivation for this specific role despite the mismatch, and long-term career aspirations. It might be more beneficial to suggest other roles within the organization or provide feedback for future applications."
        assessment_parts.append(f"**Action:** Not recommended for this role. Consider for other open positions or future pipeline, or politely decline.")

    final_assessment = f"**Overall HR Assessment: {overall_assessment_title}**\n\n"
    final_assessment += "\n".join(assessment_parts)

    return final_assessment

# Modified semantic_score to accept pre-computed embeddings
# Removed @st.cache_data as it will be called differently
def semantic_score_calculation(jd_embedding, resume_embedding, years_exp, cgpa, weighted_keyword_overlap_score, _ml_model):
    score = 0.0
    semantic_similarity = cosine_similarity(jd_embedding.reshape(1, -1), resume_embedding.reshape(1, -1))[0][0]
    semantic_similarity = float(np.clip(semantic_similarity, 0, 1))

    if _ml_model is None:
        print("DEBUG: ML model not loaded in semantic_score_calculation. Providing basic score and generic feedback.")
        basic_score = (weighted_keyword_overlap_score * 0.7)
        basic_score += min(years_exp * 5, 30)
        
        if cgpa is not None:
            if cgpa >= 3.5:
                basic_score += 5
            elif cgpa < 2.5:
                basic_score -= 5
        
        score = round(min(basic_score, 100), 2)
        
        return score, round(semantic_similarity, 2)

    try:
        years_exp_for_model = float(years_exp) if years_exp is not None else 0.0
        features = np.concatenate([jd_embedding, resume_embedding, [years_exp_for_model], [weighted_keyword_overlap_score]])
        predicted_score = _ml_model.predict([features])[0]

        blended_score = (predicted_score * 0.6) + \
                        (weighted_keyword_overlap_score * 0.1) + \
                        (semantic_similarity * 100 * 0.3)

        if semantic_similarity > 0.7 and years_exp >= 3:
            blended_score += 5
        
        if cgpa is not None:
            if cgpa >= 3.5:
                blended_score += 3
            elif cgpa >= 3.0:
                blended_score += 1
            elif cgpa < 2.5:
                blended_score -= 2

        score = float(np.clip(blended_score, 0, 100))
        
        return round(score, 2), round(semantic_similarity, 2)

    except Exception as e:
        print(f"ERROR: Error during semantic score calculation: {e}")
        traceback.print_exc()
        basic_score = (weighted_keyword_overlap_score * 0.7)
        basic_score += min(years_exp * 5, 30)
        
        if cgpa is not None:
            basic_score += 5 if cgpa >= 3.5 else (-5 if cgpa < 2.5 else 0)

        score = round(min(basic_score, 100), 2)

        return score, 0.0

def create_mailto_link(recipient_email, candidate_name, job_title="Job Opportunity", sender_name="Recruiting Team"):
    subject = urllib.parse.quote(f"Invitation for Interview - {job_title} - {candidate_name}")
    body = urllib.parse.quote(f"""Dear {candidate_name},

We were very impressed with your profile and would like to invite you for an interview for the {job_title} position.

Best regards,

The {sender_name}""")
    return f"mailto:{recipient_email}?subject={subject}&body={body}"

@st.cache_data
def generate_certificate_pdf(html_content):
    """Converts HTML content to PDF bytes."""
    try:
        pdf_bytes = HTML(string=html_content).write_pdf()
        return pdf_bytes
    except Exception as e:
        st.error(f"❌ Failed to generate PDF certificate: {e}")
        return None

def send_certificate_email(recipient_email, candidate_name, score, certificate_pdf_content, gmail_address, gmail_app_password):
    if not gmail_address or not gmail_app_password:
        st.error("❌ Email sending is not configured. Please ensure your Gmail address and App Password secrets are set in Streamlit.")
        return False

    msg = MIMEMultipart('mixed')
    msg['Subject'] = f"🎉 You've Earned It! Here's Your Certification from ScreenerPro"
    msg['From'] = gmail_address
    msg['To'] = recipient_email

    plain_text_body = f"""Hi {candidate_name},

Congratulations on successfully clearing the ScreenerPro resume screening process with a score of {score:.1f}%!

We’re proud to award you an official certificate recognizing your skills and employability.

You can add this to your resume, LinkedIn, or share it with employers to stand out.

Have questions? Contact us at support@screenerpro.in

🚀 Keep striving. Keep growing.

– Team ScreenerPro
"""

    html_body = f"""
    <html>
        <body>
            <p>Hi {candidate_name},</p>
            <p>Congratulations on successfully clearing the ScreenerPro resume screening process with a score of <strong>{score:.1f}%</strong>!</p>
            <p>We’re proud to award you an official certificate recognizing your skills and employability.</p>
            <p>You can add this to your resume, LinkedIn, or share it with employers to stand out.</p>
            <p>Have questions? Contact us at support@screenerpro.in</p>
            <p>🚀 Keep striving. Keep growing.</p>
            <p>– Team ScreenerPro</p>
        </body>
    </html>
    """

    msg_alternative = MIMEMultipart('alternative')
    msg_alternative.attach(MIMEText(plain_text_body, 'plain'))
    msg_alternative.attach(MIMEText(html_body, 'html'))
    
    msg.attach(msg_alternative)

    if certificate_pdf_content:
        try:
            attachment = MIMEBase('application', 'pdf')
            attachment.set_payload(certificate_pdf_content)
            encoders.encode_base64(attachment)
            attachment.add_header('Content-Disposition', 'attachment', filename=f'ScreenerPro_Certificate_{candidate_name.replace(" ", "_")}.pdf')
            msg.attach(attachment)
            st.info(f"Attached certificate PDF to email for {candidate_name}.")
        except Exception as e:
            st.error(f"Failed to attach certificate PDF: {e}")
    else:
        st.warning("No PDF content generated to attach to email.")

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(gmail_address, gmail_app_password)
            smtp.send_message(msg)
        st.success(f"✅ Certificate email sent to {recipient_email}!")
        return True
    except smtplib.SMTPAuthenticationError:
        st.error("❌ Failed to send email: Authentication error. Please check your Gmail address and App Password.")
        st.info("Ensure you have generated an App Password for your Gmail account and used it instead of your regular password.")
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")
    return False

# Wrapper for extract_text_from_file to be used with ProcessPoolExecutor
def _extract_text_wrapper(file_info):
    file_data_bytes, file_name, file_type = file_info
    text = extract_text_from_file(file_data_bytes, file_name, file_type)
    return file_name, text

# Modified _process_single_resume_for_screener_page
def _process_single_resume_for_screener_page(file_name, text, jd_text, jd_embedding, 
                                             resume_embedding, jd_name_for_results,
                                             high_priority_skills, medium_priority_skills, max_experience,
                                             _global_ml_model):
    """
    Processes a single resume (pre-extracted text and pre-computed embeddings)
    for the main screener page and returns a dictionary of results.
    This function is designed to be run in a ProcessPoolExecutor.
    """
    try:
        if text.startswith("[ERROR]"):
            return {
                "File Name": file_name,
                "Candidate Name": file_name.replace('.pdf', '').replace('.jpg', '').replace('.jpeg', '').replace('.png', '').replace('_', ' ').title(),
                "Score (%)": 0, "Years Experience": 0, "CGPA (4.0 Scale)": None,
                "Email": "Not Found", "Phone Number": "Not Found", "Location": "Not Found",
                "Languages": "Not Found", "Education Details": "Not Found",
                "Work History": "Not Found", "Project Details": "Not Found",
                "AI Suggestion": f"Error: {text.replace('[ERROR] ', '')}",
                "Detailed HR Assessment": f"Error processing resume: {text.replace('[ERROR] ', '')}",
                "Matched Keywords": "", "Missing Skills": "",
                "Matched Keywords (Categorized)": {}, "Missing Skills (Categorized)": {},
                "Semantic Similarity": 0.0, "Resume Raw Text": "",
                "JD Used": jd_name_for_results, "Date Screened": datetime.now().date(),
                "Certificate ID": str(uuid.uuid4()), "Certificate Rank": "Not Applicable",
                "Tag": "❌ Text Extraction Error"
            }

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

        candidate_name = extract_name(text) or file_name.replace('.pdf', '').replace('.jpg', '').replace('.jpeg', '').replace('.png', '').replace('_', ' ').title()
        cgpa = extract_cgpa(text)

        resume_raw_skills_set, resume_categorized_skills = extract_relevant_keywords(text, MASTER_SKILLS)
        jd_raw_skills_set, jd_categorized_skills = extract_relevant_keywords(jd_text, MASTER_SKILLS)

        matched_keywords = list(resume_raw_skills_set.intersection(jd_raw_skills_set))
        missing_skills = list(jd_raw_skills_set.difference(resume_raw_skills_set)) 

        # Calculate weighted keyword overlap score
        weighted_keyword_overlap_score = 0
        total_jd_skill_weight = 0
        WEIGHT_HIGH = 3
        WEIGHT_MEDIUM = 2
        WEIGHT_BASE = 1

        for jd_skill in jd_raw_skills_set:
            current_weight = WEIGHT_BASE
            if jd_skill in [s.lower() for s in high_priority_skills]:
                current_weight = WEIGHT_HIGH
            elif jd_skill in [s.lower() for s in medium_priority_skills]:
                current_weight = WEIGHT_MEDIUM
            
            total_jd_skill_weight += current_weight
            
            if jd_skill in resume_raw_skills_set:
                weighted_keyword_overlap_score += current_weight

        # Call the semantic score calculation with pre-computed embeddings
        score, semantic_similarity = semantic_score_calculation(
            jd_embedding, resume_embedding, exp, cgpa, weighted_keyword_overlap_score, _global_ml_model
        )
        
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
        
        # Determine Tag
        tag = "❌ Limited Match"
        if score >= 90 and exp >= 5 and exp <= max_experience and semantic_similarity >= 0.85 and (cgpa is None or cgpa >= 3.5):
            tag = "👑 Exceptional Match"
        elif score >= 80 and exp >= 3 and exp <= max_experience and semantic_similarity >= 0.7 and (cgpa is None or cgpa >= 3.0):
            tag = "🔥 Strong Candidate"
        elif score >= 60 and exp >= 1 and exp <= max_experience and (cgpa is None or cgpa >= 2.5):
            tag = "✨ Promising Fit"
        elif score >= 40:
            tag = "⚠️ Needs Review"

        return {
            "File Name": file_name,
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
            "Certificate ID": str(uuid.uuid4()),
            "Certificate Rank": certificate_rank,
            "Tag": tag
        }
    except Exception as e:
        print(f"CRITICAL ERROR: Unhandled exception processing {file_name}: {e}")
        traceback.print_exc()
        return {
            "File Name": file_name,
            "Candidate Name": file_name.replace('.pdf', '').replace('.jpg', '').replace('.jpeg', '').replace('.png', '').replace('_', ' ').title(),
            "Score (%)": 0, "Years Experience": 0, "CGPA (4.0 Scale)": None,
            "Email": "Not Found", "Phone Number": "Not Found", "Location": "Not Found",
            "Languages": "Not Found", "Education Details": "Not Found",
            "Work History": "Not Found", "Project Details": "Not Found",
            "AI Suggestion": f"Critical Error: {e}",
            "Detailed HR Assessment": f"Critical Error processing resume: {e}",
            "Matched Keywords": "", "Missing Skills": "",
            "Matched Keywords (Categorized)": {}, "Missing Skills (Categorized)": {},
            "Semantic Similarity": 0.0, "Resume Raw Text": "",
            "JD Used": jd_name_for_results, "Date Screened": datetime.now().date(),
            "Certificate ID": str(uuid.uuid4()), "Certificate Rank": "Not Applicable",
            "Tag": "❌ Critical Processing Error"
        }


def resume_screener_page():
    st.title("🧠 ScreenerPro – AI-Powered Resume Screener")

    if 'screening_cutoff_score' not in st.session_state:
        st.session_state['screening_cutoff_score'] = 75
    if 'screening_min_experience' not in st.session_state:
        st.session_state['screening_min_experience'] = 2
    if 'screening_max_experience' not in st.session_state:
        st.session_state['screening_max_experience'] = 10
    if 'screening_min_cgpa' not in st.session_state:
        st.session_state['screening_min_cgpa'] = 2.5
    
    if 'comprehensive_df' not in st.session_state:
        st.session_state['comprehensive_df'] = pd.DataFrame(columns=[
            "File Name", "Candidate Name", "Score (%)", "Years Experience", "CGPA (4.0 Scale)",
            "Email", "Phone Number", "Location", "Languages", "Education Details",
            "Work History", "Project Details", "AI Suggestion", "Detailed HR Assessment",
            "Matched Keywords", "Missing Skills", "Matched Keywords (Categorized)",
            "Missing Skills (Categorized)", "Semantic Similarity", "Resume Raw Text",
            "JD Used", "Date Screened", "Certificate ID", "Certificate Rank", "Tag"
        ])
    if 'resume_raw_texts' not in st.session_state:
        st.session_state['resume_raw_texts'] = {}
    if 'certificate_html_content' not in st.session_state:
        st.session_state['certificate_html_content'] = ""

    # Initial check for Tesseract (main process only)
    tesseract_cmd_path = get_tesseract_cmd()
    if not tesseract_cmd_path:
        st.error("Tesseract OCR engine not found. Please ensure it's installed and in your system's PATH.")
        st.info("On Streamlit Community Cloud, ensure you have a `packages.txt` file in your repository's root with `tesseract-ocr` and `tesseract-ocr-eng` listed.")
        st.stop()

    st.markdown("## ⚙️ Define Job Requirements & Screening Criteria")
    col1, col2 = st.columns([2, 1])

    with col1:
        jd_text = ""
        job_roles = {"Upload my own": None}
        if os.path.exists("data"):
            for fname in os.listdir("data"):
                if fname.endswith(".txt"):
                    job_roles[fname.replace(".txt", "").replace("_", " ").title()] = os.path.join("data", fname)

        jd_option = st.selectbox("📌 **Select a Pre-Loaded Job Role or Upload Your Own Job Description**", list(job_roles.keys()))
        
        jd_name_for_results = ""
        if jd_option == "Upload my own":
            jd_file = st.file_uploader("Upload Job Description (TXT, PDF)", type=["txt", "pdf"], help="Upload a .txt or .pdf file containing the job description.")
            if jd_file:
                # For JD, we read and extract text directly as it's a single file
                jd_text = extract_text_from_file(jd_file.read(), jd_file.name, jd_file.type)
                jd_name_for_results = jd_file.name.replace('.pdf', '').replace('.txt', '')
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
        cutoff = st.slider("📈 **Minimum Score Cutoff (%)**", 0, 100, 75, key="min_score_cutoff_slider", help="Candidates scoring below this percentage will be flagged for closer review or considered less suitable.")
        st.session_state['screening_cutoff_score'] = cutoff

        min_experience = st.slider("💼 **Minimum Experience Required (Years)**", 0, 15, 2, key="min_exp_slider", help="Candidates with less than this experience will be noted.")
        st.session_state['screening_min_experience'] = min_experience

        max_experience = st.slider("⬆️ **Maximum Experience Allowed (Years)**", 0, 20, 10, key="max_exp_slider", help="Candidates with more than this experience might be considered overqualified or outside the target range.")
        st.session_state['screening_max_experience'] = max_experience

        min_cgpa = st.slider("🎓 **Minimum CGPA Required (4.0 Scale)**", 0.0, 4.0, 2.5, 0.1, key="min_cgpa_slider", help="Candidates with CGPA below this value (normalized to 4.0) will be noted.")
        st.session_state['screening_min_cgpa'] = min_cgpa

        st.markdown("---")
        st.info("Once criteria are set, upload resumes below to begin screening.")

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

    resume_files = st.file_uploader("📄 **Upload Resumes (PDF, JPG, PNG)**", type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=True, help="Upload one or more PDF or image resumes for screening.")

    if jd_text and resume_files:
        # Start overall timer
        total_screening_start_time = time.time()

        st.markdown("---")
        st.markdown("## ☁️ Job Description Keyword Cloud")
        st.caption("Visualizing the most frequent and important keywords from the Job Description.")
        st.info("💡 To filter candidates by these skills, use the 'Filter Candidates by Skill' section below the main results table.")
        
        jd_words_for_cloud_set, _ = extract_relevant_keywords(jd_text, all_master_skills)
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

        total_resumes = len(resume_files)
        
        # --- PHASE 1: Parallel Text Extraction ---
        start_time_extraction = time.time()
        st.info(f"Step 1/3: Extracting text from {total_resumes} resumes concurrently...")
        extracted_texts_info = [] # Stores (file_name, text) tuples
        file_infos_for_extraction = []
        for file in resume_files:
            file_data_bytes = file.read() # Read file content into memory once
            file_infos_for_extraction.append((file_data_bytes, file.name, file.type))

        # Define a chunk size for processing
        CHUNK_SIZE = 10 # Process 10 resumes at a time to manage memory and CPU usage

        # Use ProcessPoolExecutor for CPU-bound text extraction
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor: 
            for i in range(0, total_resumes, CHUNK_SIZE):
                chunk_files_info = file_infos_for_extraction[i:i + CHUNK_SIZE]
                text_futures = [executor.submit(_extract_text_wrapper, info) for info in chunk_files_info]
                
                for j, future in enumerate(as_completed(text_futures)):
                    current_processed = i + j + 1
                    status_text.text(f"Extracting text: Processing resume {current_processed} of {total_resumes}...")
                    try:
                        extracted_texts_info.append(future.result())
                    except Exception as e:
                        st.error(f"Error extracting text for {chunk_files_info[j][1]}: {e}")
                        extracted_texts_info.append((chunk_files_info[j][1], f"[ERROR] {e}")) # Mark as error
                    progress_bar.progress(current_processed / total_resumes)
        
        end_time_extraction = time.time()
        print(f"Time taken for Text Extraction: {end_time_extraction - start_time_extraction:.2f} seconds")

        progress_bar.empty()
        status_text.empty()

        # Separate successfully extracted texts from failed ones
        successfully_extracted_texts_map = {name: text for name, text in extracted_texts_info if not text.startswith("[ERROR]")}
        failed_extraction_results = [{
            "File Name": name,
            "Candidate Name": name.replace('.pdf', '').replace('.jpg', '').replace('.jpeg', '').replace('.png', '').replace('_', ' ').title(),
            "Score (%)": 0, "Years Experience": 0, "CGPA (4.0 Scale)": None,
            "Email": "Not Found", "Phone Number": "Not Found", "Location": "Not Found",
            "Languages": "Not Found", "Education Details": "Not Found",
            "Work History": "Not Found", "Project Details": "Not Found",
            "AI Suggestion": f"Error: {text.replace('[ERROR] ', '')}",
            "Detailed HR Assessment": f"Error processing resume: {text.replace('[ERROR] ', '')}",
            "Matched Keywords": "", "Missing Skills": "",
            "Matched Keywords (Categorized)": {}, "Missing Skills (Categorized)": {},
            "Semantic Similarity": 0.0, "Resume Raw Text": "",
            "JD Used": jd_name_for_results, "Date Screened": datetime.now().date(),
            "Certificate ID": str(uuid.uuid4()), "Certificate Rank": "Not Applicable",
            "Tag": "❌ Text Extraction Error"
        } for name, text in extracted_texts_info if text.startswith("[ERROR]")]

        if not successfully_extracted_texts_map:
            st.warning("No resumes had readable text extracted. Please check the files and try again.")
            st.session_state['comprehensive_df'] = pd.DataFrame()
            return
        
        # --- PHASE 2: Batch Embedding Generation ---
        start_time_embedding = time.time()
        st.info(f"Step 2/3: Generating embeddings for {len(successfully_extracted_texts_map)} resumes and JD...")
        jd_clean = clean_text(jd_text)
        jd_embedding = global_sentence_model.encode([jd_clean])[0]

        resume_names_for_embedding = list(successfully_extracted_texts_map.keys())
        resume_texts_for_embedding = [successfully_extracted_texts_map[name] for name in resume_names_for_embedding]
        
        # Use batch_size for encoding all resume texts
        # Increased batch_size from 64 to 128 for better performance
        resume_embeddings_array = global_sentence_model.encode(
            resume_texts_for_embedding, 
            batch_size=128, # Optimized batch size
            show_progress_bar=False # Streamlit handles progress bar
        )
        
        # Create a mapping from file_name to its embedding
        resume_embedding_map = {name: embed for name, embed in zip(resume_names_for_embedding, resume_embeddings_array)}
        
        end_time_embedding = time.time()
        print(f"Time taken for Embedding Generation: {end_time_embedding - start_time_embedding:.2f} seconds")

        progress_bar.empty()
        status_text.empty()

        # --- PHASE 3: Parallel Individual Resume Analysis ---
        start_time_analysis = time.time()
        st.info(f"Step 3/3: Processing {len(successfully_extracted_texts_map)} resumes with AI models concurrently...")
        
        # Prepare arguments for the process pool
        processing_args = []
        for file_name in resume_names_for_embedding:
            text = successfully_extracted_texts_map[file_name]
            resume_embedding = resume_embedding_map[file_name]
            processing_args.append((
                file_name, text, jd_text, jd_embedding, resume_embedding,
                jd_name_for_results, high_priority_skills, medium_priority_skills, max_experience,
                global_ml_model
            ))
        
        total_successful_resumes = len(processing_args)
        current_analysis_processed = 0

        # Use ProcessPoolExecutor for CPU-bound analysis
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor: 
            for i in range(0, total_successful_resumes, CHUNK_SIZE):
                chunk_processing_args = processing_args[i:i + CHUNK_SIZE]
                analysis_futures = [executor.submit(_process_single_resume_for_screener_page, *args) for args in chunk_processing_args]
                
                for j, future in enumerate(as_completed(analysis_futures)):
                    current_analysis_processed = i + j + 1
                    status_text.text(f"Analyzing resumes: Processing candidate {current_analysis_processed} of {total_successful_resumes}...")
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as exc:
                        st.error(f"Resume processing generated an exception for {chunk_processing_args[j][0]}: {exc}")
                    progress_bar.progress(current_analysis_processed / total_successful_resumes)
        
        # Add results from failed extractions back to the list
        results.extend(failed_extraction_results)

        end_time_analysis = time.time()
        print(f"Time taken for Individual Resume Analysis: {end_time_analysis - start_time_analysis:.2f} seconds")

        progress_bar.empty()
        status_text.empty()
        
        if not results:
            st.warning("No resumes were successfully processed. Please check the files and try again.")
            st.session_state['comprehensive_df'] = pd.DataFrame()
            return

        st.session_state['comprehensive_df'] = pd.DataFrame(results).sort_values(by="Score (%)", ascending=False).reset_index(drop=True)
        
        st.session_state['comprehensive_df'].to_csv("results.csv", index=False)

        total_screening_end_time = time.time()
        print(f"Total time for entire screening process: {total_screening_end_time - total_screening_start_time:.2f} seconds")


        st.markdown("---")
        st.markdown("## 📊 Candidate Score Comparison")
        st.caption("Visual overview of how each candidate ranks against the job requirements.")
        dark_mode = st.session_state.get("dark_mode_main", False)

        if not st.session_state['comprehensive_df'].empty:
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
                        # Generate HTML content for the certificate (for preview and PDF conversion)
                        certificate_html_content = generate_certificate_html(candidate_data_for_cert)
                        st.session_state['certificate_html_content'] = certificate_html_content # Store for preview

                        # Generate PDF content
                        certificate_pdf_content = generate_certificate_pdf(certificate_html_content)

                        col_cert_view, col_cert_download = st.columns(2)
                        with col_cert_view:
                            if st.button("👁️ View Certificate (HTML Preview)", key="view_cert_button"):
                                # This button just triggers the preview, content is already generated
                                pass 
                                
                        with col_cert_download:
                            if certificate_pdf_content:
                                st.download_button(
                                    label="⬇️ Download Certificate (PDF)",
                                    data=certificate_pdf_content,
                                    file_name=f"ScreenerPro_Certificate_{candidate_data_for_cert['Candidate Name'].replace(' ', '_')}.pdf",
                                    mime="application/pdf",
                                    key="download_cert_pdf_button"
                                )
                            else:
                                st.warning("PDF generation failed, cannot provide download.")
                        
                        # Automatically send email if certificate PDF is generated and email is available
                        if candidate_data_for_cert.get('Email') and candidate_data_for_cert['Email'] != "Not Found":
                            if certificate_pdf_content:
                                gmail_address = st.secrets.get("GMAIL_ADDRESS")
                                gmail_app_password = st.secrets.get("GMAIL_APP_PASSWORD")
                                send_certificate_email(
                                    recipient_email=candidate_data_for_cert['Email'],
                                    candidate_name=candidate_data_for_cert['Candidate Name'],
                                    score=candidate_data_for_cert['Score (%)'],
                                    certificate_pdf_content=certificate_pdf_content,
                                    gmail_address=gmail_address,
                                    gmail_app_password=gmail_app_password
                                )
                            else:
                                st.error("PDF certificate could not be generated, so email could not be sent.")
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
        st.markdown("### Generated Certificate Preview (HTML)")
        st.components.v1.html(st.session_state['certificate_html_content'], height=600, scrolling=True)
        st.markdown("---")


    else:
        st.info("Please upload a Job Description and at least one Resume to begin the screening process.")

@st.cache_data
def generate_certificate_html(candidate_data):
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ScreenerPro Certification - Candidate Name</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Playfair+Display:wght@700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f8f9fa; /* Light background for print compatibility */
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
            box-sizing: border-box;
        }
        .certificate-container {
            width: 100%;
            max-width: 800px;
            background: linear-gradient(145deg, #ffffff, #e6e6e6);
            border: 10px solid #00cec9; /* Main border color */
            border-radius: 20px;
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.2);
            padding: 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        .logo-text {
            font-family: 'Playfair Display', serif; /* A more elegant font for the logo */
            font-size: 2.8em; /* Larger logo text */
            color: #00cec9;
            font-weight: 700;
            margin-bottom: 15px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .certificate-header {
            font-family: 'Playfair Display', serif;
            font-size: 2.5em;
            color: #34495e; /* Darker text for header */
            margin-bottom: 20px;
            font-weight: 700;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
        }
        .certificate-subheader {
            font-size: 1.2em;
            color: #555;
            margin-bottom: 30px;
        }
        .certificate-body {
            margin-bottom: 40px;
        }
        .certificate-body p {
            font-size: 1.1em;
            line-height: 1.6;
            color: #333;
        }
        .candidate-name {
            font-family: 'Playfair Display', serif;
            font-size: 3.2em; /* Even larger name */
            color: #00cec9; /* Teal for the name */
            margin: 25px 0;
            font-weight: 700;
            border-bottom: 3px dashed #b0e0e6; /* Lighter dashed line */
            display: inline-block;
            padding-bottom: 8px;
            animation: pulse 1.5s infinite alternate; /* Subtle animation */
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            100% { transform: scale(1.02); }
        }
        .score-rank {
            font-size: 1.6em; /* Larger score/rank */
            color: #28a745; /* Green for success */
            font-weight: 700;
            margin-top: 20px;
            padding: 5px 15px;
            background-color: #e6ffe6; /* Light green background */
            border-radius: 10px;
            display: inline-block;
        }
        .date-id {
            font-size: 0.95em;
            color: #777;
            margin-top: 35px;
        }
        .footer-text {
            font-size: 0.85em;
            color: #999;
            margin-top: 40px;
        }
        /* Print styles */
        @media print {
            body {
                background-color: #fff;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }
            .certificate-container {
                border: 5px solid #00cec9;
                box-shadow: none;
            }
            .logo-text, .certificate-header, .candidate-name, .score-rank {
                color: #00cec9 !important; /* Ensure colors are printed */
                text-shadow: none !important;
            }
            .candidate-name {
                animation: none !important; /* Disable animation for print */
            }
        }
    </style>
</head>
<body>
    <div class="certificate-container">
        <div class="logo-text">ScreenerPro</div>
        <div class="certificate-header">Certification of Excellence</div>
        <div class="certificate-subheader">This is to proudly certify that</div>
        <div class="candidate-name">{{CANDIDATE_NAME}}</div>
        <div class="certificate-body">
            <p>has successfully completed the comprehensive AI-powered screening process by ScreenerPro and achieved a distinguished ranking of</p>
            <div class="score-rank">
                {{CERTIFICATE_RANK}}
            </div>
            <p>demonstrating an outstanding Screener Score of **{{SCORE}}%**.</p>
            <p>This certification attests to their highly relevant skills, extensive experience, and strong alignment with the demanding requirements of modern professional roles. It signifies their readiness to excel in challenging environments and contribute significantly to organizational success.</p>
        </div>
        <div class="date-id">
            Awarded on: {{DATE_SCREENED}}<br>
            Certificate ID: {{CERTIFICATE_ID}}
        </div>
        <div class="footer-text">
            This certificate is digitally verified by ScreenerPro.
        </div>
    </div>
</body>
</html>
    """

    candidate_name = candidate_data.get('Candidate Name', 'Candidate Name')
    score = candidate_data.get('Score (%)', 0.0)
    certificate_rank = candidate_data.get('Certificate Rank', 'Not Applicable')
    date_screened = candidate_data.get('Date Screened', datetime.now().date()).strftime("%B %d, %Y")
    certificate_id = candidate_data.get('Certificate ID', 'N/A')
    
    html_content = html_template.replace("{{CANDIDATE_NAME}}", candidate_name)
    html_content = html_content.replace("{{SCORE}}", f"{score:.1f}")
    html_content = html_content.replace("{{CERTIFICATE_RANK}}", certificate_rank)
    html_content = html_content.replace("{{DATE_SCREENED}}", date_screened)
    html_content = html_content.replace("{{CERTIFICATE_ID}}", certificate_id)

    return html_content

if __name__ == "__main__":
    resume_screener_page()
