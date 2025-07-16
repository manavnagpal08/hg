import streamlit as st
import pdfplumber
import pandas as pd
import re
import os
import sklearn
import joblib
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
import nltk
import collections
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse
import uuid # Added for generating unique IDs
import json # For handling JSON data for Firestore

# --- OCR Specific Imports ---
from PIL import Image
import pytesseract
import io

# --- Tesseract OCR Path Configuration ---
# IMPORTANT: You MUST change this path to your Tesseract executable.
# For Windows, it might look like: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# For macOS/Linux, it's often in your PATH, so you might not need this line
# or it could be '/usr/local/bin/tesseract'
# If Tesseract is not installed or path is incorrect, OCR will fail.
try:
    # Attempt to set the Tesseract command path. This might not be necessary on all OS.
    # If you get a TesseractNotFoundError, uncomment the line below and set the correct path.
    # pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract' # Example for macOS/Linux
    pass
except Exception as e:
    st.warning(f"⚠️ Could not set Tesseract command path automatically: {e}. Please ensure Tesseract is installed and its path is correctly configured in screener.py if you encounter OCR issues.")


# Download NLTK stopwords data if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- Load Embedding + ML Model ---
@st.cache_resource
def load_ml_model():
    """
    Loads the SentenceTransformer model for embeddings and the pre-trained ML screening model.
    Uses st.cache_resource to cache the models for efficient reuse.
    """
    try:
        # Load the SentenceTransformer model for generating embeddings
        # This model converts text into numerical vectors that capture semantic meaning.
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Load the pre-trained machine learning model (e.g., a classifier)
        # This model is used for the actual resume screening/prediction.
        ml_model = joblib.load("ml_screening_model.pkl")
        return model, ml_model
    except Exception as e:
        # Display an error message if models fail to load, guiding the user to check files.
        st.error(f"❌ Error loading ML models: {e}. Please ensure 'ml_screening_model.pkl' is in the same directory and SentenceTransformer can download its components.")
        return None, None

# Load models at the start of the script
model, ml_model = load_ml_model()

# --- Predefined List of Cities for Location Extraction ---
# This list helps in accurately identifying locations mentioned in resumes.
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


# --- Stop Words List (Using NLTK and Custom Additions) ---
# Stop words are common words (e.g., "the", "is") that are usually removed
# before processing text to focus on more meaningful terms.
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
# Combine NLTK and custom stop words for a comprehensive list
STOP_WORDS = NLTK_STOP_WORDS.union(CUSTOM_STOP_WORDS)

# --- Skill Categories (for categorization and weighting) ---
# This dictionary helps in organizing and identifying skills based on their domain.
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

# Dynamically generate MASTER_SKILLS from SKILL_CATEGORIES
MASTER_SKILLS = set([skill.lower() for category_list in SKILL_CATEGORIES.values() for skill in category_list])


# --- Helpers ---
def clean_text(text):
    """
    Cleans text by removing newlines, extra spaces, and non-ASCII characters.
    Converts text to lowercase for consistent processing.
    """
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip().lower()

def extract_relevant_keywords(text, filter_set):
    """
    Extracts relevant keywords from text, prioritizing multi-word skills from filter_set.
    If filter_set is empty, it falls back to filtering out general STOP_WORDS.
    Returns a tuple: (raw_keywords_set, categorized_keywords_dict)
    """
    cleaned_text = clean_text(text)
    extracted_keywords = set()
    categorized_keywords = collections.defaultdict(list)

    if filter_set: # If a specific filter_set (like MASTER_SKILLS) is provided
        # Sort skills by length descending to match longer phrases first
        sorted_filter_skills = sorted(list(filter_set), key=len, reverse=True)
        
        temp_text = cleaned_text # Use a temporary text to remove matched phrases

        for skill_phrase in sorted_filter_skills:
            # Create a regex pattern to match the whole skill phrase
            # \b ensures whole word match, re.escape handles special characters in skill names
            pattern = r'\b' + re.escape(skill_phrase.lower()) + r'\b'
            
            # Find all occurrences of the skill phrase
            matches = re.findall(pattern, temp_text)
            if matches:
                # Add the original skill (title-cased for better display)
                extracted_keywords.add(skill_phrase.title()) 
                # Categorize the skill
                found_category = False
                for category, skills_in_category in SKILL_CATEGORIES.items():
                    if skill_phrase.lower() in [s.lower() for s in skills_in_category]:
                        categorized_keywords[category].append(skill_phrase.title())
                        found_category = True
                        break
                if not found_category:
                    categorized_keywords["Uncategorized"].append(skill_phrase.title()) # Add to uncategorized if no match

                # Replace the found skill with placeholders to avoid re-matching parts of it
                temp_text = re.sub(pattern, " ", temp_text)
        
        # After extracting phrases, now extract individual words that are in the filter_set
        # and haven't been part of a multi-word skill already extracted.
        # This ensures single-word skills from MASTER_SKILLS are also captured.
        individual_words_remaining = set(re.findall(r'\b\w+\b', temp_text))
        for word in individual_words_remaining:
            if word in filter_set: # Check if the single word is in the master skills
                extracted_keywords.add(word.title()) # Add the original word (title-cased)
                found_category = False
                for category, skills_in_category in SKILL_CATEGORIES.items():
                    if word.lower() in [s.lower() for s in skills_in_category]:
                        categorized_keywords[category].append(word.title())
                        found_category = True
                        break
                if not found_category:
                    categorized_keywords["Uncategorized"].append(word.title())

    else: # Fallback: if no specific filter_set (MASTER_SKILLS is empty), use the default STOP_WORDS logic
        all_words = set(re.findall(r'\b\w+\b', cleaned_text))
        extracted_keywords = {word for word in all_words if word not in STOP_WORDS}
        for word in extracted_keywords:
            categorized_keywords["General Keywords"].append(word.title()) # Default category for fallback

    return extracted_keywords, dict(categorized_keywords)

def extract_text_from_image_pdf(uploaded_file):
    """
    Extracts text from a PDF file by performing OCR on each page.
    This is used as a fallback if pdfplumber fails to extract sufficient text,
    indicating a scanned (image-based) PDF.
    """
    full_text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Render the page as an image
                page_image = page.to_image()
                
                # Convert PIL Image to bytes for Tesseract
                img_byte_arr = io.BytesIO()
                page_image.original.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()

                # Perform OCR using pytesseract
                text_from_page = pytesseract.image_to_string(Image.open(io.BytesIO(img_byte_arr)))
                full_text += text_from_page + "\n" # Add newline between pages
        return full_text
    except pytesseract.TesseractNotFoundError:
        st.error("Tesseract OCR is not installed or not found in your PATH. Please install it to enable OCR for scanned PDFs.")
        return "[ERROR] Tesseract not found."
    except Exception as e:
        st.error(f"Error during OCR extraction: {e}")
        return f"[ERROR] {str(e)}"

def extract_text_from_pdf(uploaded_file):
    """
    Extracts text from an uploaded PDF file.
    First tries pdfplumber for text-based PDFs.
    If little text is extracted, falls back to OCR for scanned PDFs.
    """
    # Try pdfplumber first (faster for text-based PDFs)
    text_from_pdfplumber = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            text_from_pdfplumber = ''.join(page.extract_text() or '' for page in pdf.pages)
    except Exception as e:
        st.warning(f"PDFPlumber failed (might be scanned PDF): {e}. Attempting OCR...")
        text_from_pdfplumber = "" # Clear in case of partial failure

    # If pdfplumber extracted very little text, try OCR
    # Threshold can be adjusted based on typical resume length
    if len(text_from_pdfplumber.strip()) < 100: # Heuristic: if less than 100 characters, try OCR
        st.info("Falling back to OCR for text extraction (this may take longer)...")
        # Reset file pointer for the second read
        uploaded_file.seek(0) 
        text_from_ocr = extract_text_from_image_pdf(uploaded_file)
        if "[ERROR]" in text_from_ocr:
            return text_from_ocr # Return OCR error if any
        elif len(text_from_ocr.strip()) > len(text_from_pdfplumber.strip()):
            return text_from_ocr # Use OCR text if it's more substantial
        else:
            return text_from_pdfplumber # Fallback to original if OCR is worse or same
    
    return text_from_pdfplumber


def extract_years_of_experience(text):
    """
    Extracts years of experience from a given text by parsing date ranges or keywords.
    It looks for 'start_date to end_date' patterns or direct mentions of 'X years experience'.
    """
    text = text.lower()
    total_months = 0
    
    # Pattern to find date ranges like "Jan 2020 - Dec 2022" or "January 2020 to Present"
    job_date_ranges = re.findall(
        r'(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})\s*(?:to|–|-)\s*(present|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})',
        text
    )

    for start, end in job_date_ranges:
        try:
            # Attempt to parse start date
            start_date = datetime.strptime(start.strip(), '%b %Y')
        except ValueError:
            try:
                start_date = datetime.strptime(start.strip(), '%B %Y')
            except ValueError:
                continue # Skip if date format is not recognized

        if end.strip() == 'present':
            end_date = datetime.now() # Use current date for 'present'
        else:
            try:
                end_date = datetime.strptime(end.strip(), '%b %Y')
            except ValueError:
                try:
                    end_date = datetime.strptime(end.strip(), '%B %Y')
                except ValueError:
                    continue # Skip if date format is not recognized

        # Calculate difference in months
        delta_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        total_months += max(delta_months, 0) # Ensure months are not negative

    if total_months == 0:
        # Fallback: if no date ranges found, look for direct mentions of "X years experience"
        match = re.search(r'(\d+(?:\.\d+)?)\s*(\+)?\s*(year|yrs|years)\b', text)
        if not match:
            match = re.search(r'experience[^\d]{0,10}(\d+(?:\.\d+)?)', text)
        if match:
            return float(match.group(1)) # Return the directly found years

    return round(total_months / 12, 1) # Convert total months to years

def extract_email(text):
    """Extracts an email address from the given text using a regular expression."""
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else None

def extract_phone_number(text):
    """
    Extracts a phone number from the given text.
    Uses a robust regex to match various phone number formats.
    """
    # Common patterns: (XXX) XXX-XXXX, XXX-XXX-XXXX, XXXXXXXXXX, XXX.XXX.XXXX, +XX XXX XXXXXXXX
    match = re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', text)
    return match.group(0) if match else None

def extract_location(text):
    """
    Extracts location details by matching against a predefined list of cities (MASTER_CITIES).
    This approach is less reliant on external NLP models and should be more stable
    in various deployment environments.
    """
    found_locations = set()
    text_lower = text.lower()

    # Sort cities by length descending to prioritize longer, more specific matches
    # (e.g., "New York City" before "New York")
    sorted_cities = sorted(list(MASTER_CITIES), key=len, reverse=True)

    for city in sorted_cities:
        # Use regex to find whole word matches for the city name
        # re.escape handles special characters in city names (e.g., "St. Louis")
        pattern = r'\b' + re.escape(city.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found_locations.add(city) # Add the original, title-cased city name

    if found_locations:
        return ", ".join(sorted(list(found_locations)))
    return "Not Found"


def extract_name(text):
    """
    Attempts to extract a name from the first few lines of the resume text.
    This is a heuristic and might not be perfect for all resume formats.
    Filters out common non-name terms like "LinkedIn".
    """
    lines = text.strip().split('\n')
    if not lines:
        return None

    # Define terms to explicitly exclude from being identified as a name
    EXCLUDE_NAME_TERMS = {"linkedin", "github", "portfolio", "resume", "cv", "profile", "contact", "email", "phone", "education", "experience", "skills", "projects", "certifications"}

    potential_name_lines = []
    # Consider the first 5-10 lines for name extraction as names are usually at the top
    for line in lines[:10]:
        line = line.strip()
        line_lower = line.lower()

        # Filter out lines that clearly contain email, phone, or too many words
        # Also, filter out lines that contain any of the EXCLUDE_NAME_TERMS
        if not re.search(r'[@\d\.\-]', line) and \
           len(line.split()) <= 4 and \
           not any(term in line_lower for term in EXCLUDE_NAME_TERMS):
            # Heuristic: Check if the line is mostly capitalized words (like a name)
            # or if it's a mix of capitalized words which is common for names.
            if line.isupper() or (line and line[0].isupper() and all(word[0].isupper() or not word.isalpha() for word in line.split() if word.isalpha())):
                potential_name_lines.append(line)

    if potential_name_lines:
        # Prioritize longer potential names, then filter out common resume section headers
        # Use a more robust way to select the name, preferring lines with multiple capitalized words
        best_name = ""
        max_capitalized_words = 0
        for name_candidate in potential_name_lines:
            capitalized_words_count = sum(1 for word in name_candidate.split() if word and word[0].isupper())
            if capitalized_words_count > max_capitalized_words:
                best_name = name_candidate
                max_capitalized_words = capitalized_words_count
            elif capitalized_words_count == max_capitalized_words and len(name_candidate) > len(best_name):
                best_name = name_candidate
        
        name = re.sub(r'summary|education|experience|skills|projects|certifications|profile|contact', '', best_name, flags=re.IGNORECASE).strip()
        # Further clean up any leading/trailing non-alphabetic characters if they remain
        name = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', name).strip()
        if name:
            return name.title() # Return name in Title Case
    return None

def extract_cgpa(text):
    """
    Extracts CGPA/GPA from text. Handles formats like X.X/Y.Y, X.X out of Y.Y, or just X.X.
    Assumes a standard scale (e.g., 4.0, 5.0, 10.0).
    Returns the CGPA normalized to a 4.0 scale if a scale is found, otherwise the raw value.
    """
    text = text.lower()
    
    # Regex to find patterns like "3.5/4.0", "3.8/5", "8.5/10", "gpa 3.7", "cgpa of 3.9"
    # Group 1: CGPA value, Group 2: Optional scale (e.g., /4.0, out of 5)
    matches = re.findall(r'(?:cgpa|gpa|grade point average)\s*[:\s]*(\d+\.\d+)(?:\s*[\/of]{1,4}\s*(\d+\.\d+|\d+))?|(\d+\.\d+)(?:\s*[\/of]{1,4}\s*(\d+\.\d+|\d+))?\s*(?:cgpa|gpa)', text)

    for match in matches:
        # Prioritize matches where 'cgpa' or 'gpa' keyword is present
        if match[0] and match[0].strip(): # First pattern: (cgpa|gpa)\s*[:\s]*(\d+\.\d+)(?:\s*[\/of]{1,4}\s*(\d+\.\d+|\d+))?
            raw_cgpa = float(match[0])
            scale = float(match[1]) if match[1] else None
        elif match[2] and match[2].strip(): # Second pattern: (\d+\.\d+)(?:\s*[\/of]{1,4}\s*(\d+\.\d+|\d+))?\s*(cgpa|gpa)
            raw_cgpa = float(match[2])
            scale = float(match[3]) if match[3] else None
        else:
            continue

        if scale and scale not in [0, 1]: # Avoid division by zero or scale of 1
            # Normalize to 4.0 scale
            normalized_cgpa = (raw_cgpa / scale) * 4.0
            return round(normalized_cgpa, 2)
        elif raw_cgpa <= 4.0: # Assume it's already on a 4.0 scale if no explicit scale and value is low
            return round(raw_cgpa, 2)
        elif raw_cgpa <= 10.0: # Assume it's on a 10.0 scale if value is higher than 4 but less than 10
            return round((raw_cgpa / 10.0) * 4.0, 2)
        
    return None # Return None if no CGPA found

def extract_education_details(text):
    """
    Extracts education details (University, Degree, Major, Year) from text.
    This is a heuristic and may require refinement based on resume formats.
    Returns a list of dicts.
    """
    education_section_matches = re.finditer(r'(?:education|academic background|qualifications)\s*(\n|$)', text, re.IGNORECASE)
    education_details = []
    
    start_index = -1
    for match in education_section_matches:
        start_index = match.end()
        break # Take the first education section

    if start_index != -1:
        # Try to find the end of the education section (e.g., start of next major section)
        sections = ['experience', 'work history', 'skills', 'projects', 'certifications', 'awards', 'publications']
        end_index = len(text)
        for section in sections:
            section_match = re.search(r'\b' + re.escape(section) + r'\b', text[start_index:], re.IGNORECASE)
            if section_match:
                end_index = start_index + section_match.start()
                break
        
        education_text = text[start_index:end_index].strip()
        
        # Split into potential education blocks (e.g., by degree or year lines)
        # Look for lines that contain a degree or a year range
        edu_blocks = re.split(r'\n(?=\s*(?:bachelor|master|phd|associate|diploma|certificat|graduat|postgraduat|doctorate|university|college|institute|school|academy)\b|\d{4}\s*[-–]\s*(?:\d{4}|present))', education_text, flags=re.IGNORECASE)
        
        for block in edu_blocks:
            block = block.strip()
            if not block:
                continue
            
            uni = None
            degree = None
            major = None
            year = None

            # Try to extract year (e.g., 2020, 2018-2022)
            year_match = re.search(r'(\d{4})\s*[-–]\s*(\d{4}|present)|\b(\d{4})\b', block)
            if year_match:
                if year_match.group(1) and year_match.group(2):
                    year = f"{year_match.group(1)}-{year_match.group(2)}"
                elif year_match.group(3):
                    year = year_match.group(3)

            # Try to extract degree (e.g., Bachelor of Science, M.S., Ph.D.)
            degree_match = re.search(r'\b(b\.?s\.?|bachelor of science|b\.?a\.?|bachelor of arts|m\.?s\.?|master of science|m\.?a\.?|master of arts|ph\.?d\.?|doctor of philosophy|mba|master of business administration|diploma|certificate)\b', block, re.IGNORECASE)
            if degree_match:
                degree = degree_match.group(0).title()

            # Try to extract university (often capitalized, common keywords)
            # This is very hard without a pre-defined list of universities
            # For now, a simple heuristic: look for capitalized phrases near "university" or "college"
            uni_match = re.search(r'\b([A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*)\s+(?:university|college|institute|school|academy)\b', block, re.IGNORECASE)
            if uni_match:
                uni = uni_match.group(1)
            else: # Fallback: look for any capitalized phrase that might be a university name
                lines = block.split('\n')
                for line in lines:
                    potential_uni_match = re.search(r'^[A-Z][a-zA-Z\s,&\.]+\b(university|college|institute|school|academy)?', line.strip())
                    if potential_uni_match and len(potential_uni_match.group(0).split()) > 1:
                        uni = potential_uni_match.group(0).strip().replace(',', '')
                        break
            
            # Try to extract major (e.g., Computer Science, Electrical Engineering)
            major_match = re.search(r'(?:in|of)\s+([A-Z][a-zA-Z\s]+(?:engineering|science|arts|business|management|studies|technology))', block, re.IGNORECASE)
            if major_match:
                major = major_match.group(1).strip()
            
            if uni or degree or major or year:
                education_details.append({
                    "University": uni,
                    "Degree": degree,
                    "Major": major,
                    "Year": year
                })
    return education_details


def extract_work_history(text):
    """
    Extracts work history details (Company, Title, Start Date, End Date) from text.
    This is a heuristic and may not capture all formats.
    Returns a list of dicts.
    """
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
        
        # Split by common job title/company patterns or date ranges.
        # This regex looks for a line starting with a capitalized word (potential company/title)
        # followed by a date range.
        job_blocks = re.split(r'\n(?=[A-Z][a-zA-Z\s,&\.]+(?:\s(?:at|@))?\s*[A-Z][a-zA-Z\s,&\.]*\s*(?:-|\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}))', work_text, flags=re.IGNORECASE)
        
        for block in job_blocks:
            block = block.strip()
            if not block:
                continue
            
            company = None
            title = None
            start_date = None
            end_date = None

            # Extract dates (e.g., Jan 2020 - Dec 2022, 2018 - Present)
            date_range_match = re.search(
                r'((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}|\d{4})\s*[-–]\s*(present|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}|\d{4})',
                block, re.IGNORECASE
            )
            if date_range_match:
                start_date = date_range_match.group(1)
                end_date = date_range_match.group(2)
                # Remove dates from block to help with company/title extraction
                block = block.replace(date_range_match.group(0), '').strip()

            # Try to extract title and company
            # Look for lines that look like "Job Title at Company Name" or "Company Name, Job Title"
            lines = block.split('\n')
            for line in lines:
                line = line.strip()
                if not line: continue

                # Pattern: "Job Title at Company Name"
                title_company_match = re.search(r'([A-Z][a-zA-Z\s,\-&.]+)\s+(?:at|@)\s+([A-Z][a-zA-Z\s,\-&.]+)', line)
                if title_company_match:
                    title = title_company_match.group(1).strip()
                    company = title_company_match.group(2).strip()
                    break
                
                # Pattern: "Company Name, Job Title" (often first line of a block)
                company_title_match = re.search(r'^([A-Z][a-zA-Z\s,\-&.]+),\s*([A-Z][a-zA-Z\s,\-&.]+)', line)
                if company_title_match:
                    company = company_title_match.group(1).strip()
                    title = company_title_match.group(2).strip()
                    break
                
                # Fallback: Just try to get a capitalized phrase as company/title
                if not company and not title:
                    potential_org_match = re.search(r'^[A-Z][a-zA-Z\s,\-&.]+', line)
                    if potential_org_match and len(potential_org_match.group(0).split()) > 1:
                        if not company: company = potential_org_match.group(0).strip()
                        elif not title: title = potential_org_match.group(0).strip()
                        break # Take the first good match

            if company or title or start_date or end_date:
                work_details.append({
                    "Company": company,
                    "Title": title,
                    "Start Date": start_date,
                    "End Date": end_date
                })
    return work_details

def extract_project_details(text):
    """
    Extracts project details (Title, Description, Technologies) from text.
    This is a heuristic and may not capture all formats.
    Returns a list of dicts.
    """
    project_section_matches = re.finditer(r'(?:projects|personal projects|key projects)\s*(\n|$)', text, re.IGNORECASE)
    project_details = []
    
    start_index = -1
    for match in project_section_matches:
        start_index = match.end()
        break

    if start_index != -1:
        sections = ['education', 'experience', 'work history', 'skills', 'certifications', 'awards', 'publications']
        end_index = len(text)
        for section in sections:
            section_match = re.search(r'\b' + re.escape(section) + r'\b', text[start_index:], re.IGNORECASE)
            if section_match:
                end_index = start_index + section_match.start()
                break
        
        project_text = text[start_index:end_index].strip()
        
        # Split into individual projects, often marked by a title line or bullet point
        # This regex tries to split by a line starting with a capital letter followed by words,
        # often indicating a new project title.
        project_blocks = re.split(r'\n(?=[A-Z][a-zA-Z\s,&\-]+\s*(?:\()?\d{4}(?:\))?)|\n(?=•\s*[A-Z][a-zA-Z\s,&\-]+)', project_text)
        
        for block in project_blocks:
            block = block.strip()
            if not block:
                continue
            
            title = None
            description = []
            technologies = []

            lines = block.split('\n')
            if lines:
                # First line is often the title
                title_line = lines[0].strip()
                if len(title_line.split()) <= 10 and re.match(r'^[A-Z]', title_line): # Heuristic for title line
                    title = title_line
                    description_lines = lines[1:]
                else:
                    description_lines = lines # If no clear title line, whole block is description
                
                # Extract technologies (simple approach: look for words in MASTER_SKILLS within the block)
                block_lower = block.lower()
                for skill in MASTER_SKILLS: # MASTER_SKILLS is now dynamically generated
                    if re.search(r'\b' + re.escape(skill.lower()) + r'\b', block_lower):
                        technologies.append(skill.title()) # Add title-cased skill
                
                # Remaining lines form the description
                description = [line.strip() for line in description_lines if line.strip()]
                
            if title or description or technologies:
                project_details.append({
                    "Project Title": title,
                    "Description": "\n".join(description),
                    "Technologies Used": ", ".join(technologies)
                })
    return project_details

def extract_languages(text):
    """
    Extracts spoken languages from the resume text.
    It primarily looks for a dedicated "Languages" or "Linguistic Abilities" section.
    If such a section is found, it extracts capitalized words from that section,
    filtering out common non-language terms. If no such section is found,
    it returns "Not Found".
    """
    languages_list = []
    text_lower = text.lower()

    # Define common section headers for languages
    language_section_headers = [
        r'\b(?:languages|linguistic abilities|language skills|proficiencies in languages)\b'
    ]

    # Define terms to exclude from being identified as languages if found in a general context
    # These are typically not languages themselves but might appear near language names.
    EXCLUDE_LANGUAGE_TERMS = {
        "fluent", "native", "proficient", "intermediate", "beginner", "basic",
        "speaking", "reading", "writing", "listening", "level", "levels", "skills",
        "ability", "abilities", "knowledge", "expertise"
    }

    language_section_text = None

    # Search for a dedicated language section
    for header_pattern in language_section_headers:
        match = re.search(header_pattern, text_lower)
        if match:
            start_index = match.end()
            # Try to find the end of the language section (e.g., start of next major section)
            sections_after_languages = ['education', 'experience', 'work history', 'skills', 'projects', 'certifications', 'awards', 'publications', 'interests', 'hobbies']
            end_index = len(text)
            for section in sections_after_languages:
                section_match = re.search(r'\b' + re.escape(section) + r'\b', text_lower[start_index:], re.IGNORECASE)
                if section_match:
                    end_index = start_index + section_match.start()
                    break
            language_section_text = text[start_index:end_index].strip()
            break # Found a language section, stop searching for headers

    if language_section_text:
        # If a language section is found, extract capitalized words (potential language names)
        # from within that section and filter out common non-language terms.
        words_in_section = re.findall(r'\b[A-Z][a-zA-Z]+\b', language_section_text)
        for word in words_in_section:
            word_lower = word.lower()
            if word_lower not in EXCLUDE_LANGUAGE_TERMS:
                languages_list.append(word.title()) # Add in title case

    # Remove duplicates and sort
    languages_list = sorted(list(set(languages_list)))

    return ", ".join(languages_list) if languages_list else "Not Found"


# --- Resume Processing Function ---
def process_resume(uploaded_file, job_description_text):
    """
    Processes an uploaded resume and a job description to extract information,
    calculate similarity, and generate insights.
    """
    if model is None or ml_model is None:
        st.error("ML models not loaded. Cannot process resume.")
        return None

    # Extract text from resume
    resume_text = extract_text_from_pdf(uploaded_file)
    if "[ERROR]" in resume_text:
        st.error(f"Failed to extract text from resume: {resume_text}")
        return None

    # --- Extract Resume Details ---
    with st.spinner("Extracting resume details..."):
        name = extract_name(resume_text)
        email = extract_email(resume_text)
        phone = extract_phone_number(resume_text)
        location = extract_location(resume_text)
        experience = extract_years_of_experience(resume_text)
        cgpa = extract_cgpa(resume_text)
        education_details = extract_education_details(resume_text)
        work_history_details = extract_work_history(resume_text)
        project_details = extract_project_details(resume_text)
        
        # Extract skills, categorized
        raw_resume_skills, categorized_resume_skills = extract_relevant_keywords(resume_text, MASTER_SKILLS)
        
        # Extract languages (now with empty master list, relies on section detection)
        extracted_languages = extract_languages(resume_text)

    # --- Process Job Description ---
    # Extract skills from JD
    jd_skills, _ = extract_relevant_keywords(job_description_text, MASTER_SKILLS)
    jd_cleaned = clean_text(job_description_text)

    # --- Calculate Similarity (using SentenceTransformer embeddings) ---
    with st.spinner("Calculating resume-JD similarity..."):
        resume_embedding = model.encode(clean_text(resume_text), convert_to_tensor=True)
        jd_embedding = model.encode(jd_cleaned, convert_to_tensor=True)
        
        # Cosine similarity between resume and JD embeddings
        similarity_score = cosine_similarity(resume_embedding.reshape(1, -1), jd_embedding.reshape(1, -1))[0][0]
        # Normalize score to 0-100 scale for easier interpretation
        normalized_similarity = (similarity_score + 1) / 2 * 100 # Adjust from -1 to 1 range to 0 to 100
        
        # Calculate skill match percentage
        matching_skills = raw_resume_skills.intersection(jd_skills)
        skill_match_percentage = (len(matching_skills) / len(jd_skills) * 100) if jd_skills else 0

    # --- Predict Screening Outcome (using pre-trained ML model) ---
    with st.spinner("Predicting screening outcome..."):
        # Create a feature vector for the ML model.
        # This is a simplified example; a real model would need more structured features.
        # For demonstration, let's use experience, CGPA, and similarity as features.
        # Ensure features are in the format expected by your ml_screening_model.pkl
        # The model was likely trained on a specific set of features.
        
        # Placeholder for feature vector - adjust this based on your actual model's training
        # For a simple demo, let's create a dummy feature array.
        # In a real scenario, you'd need to ensure the features match those used for training.
        # Example: features = np.array([[experience, cgpa, normalized_similarity, len(matching_skills)]])
        
        # If your model was trained on embeddings, you might directly use embeddings.
        # If it was trained on extracted features, you need to reconstruct those.
        
        # For now, let's assume the model expects a single embedding or a simple feature set.
        # If ml_screening_model.pkl is a classifier on embeddings, you'd do:
        # prediction = ml_model.predict(resume_embedding.cpu().numpy().reshape(1, -1))[0]
        # prediction_proba = ml_model.predict_proba(resume_embedding.cpu().numpy().reshape(1, -1))[0]
        
        # Assuming ml_screening_model.pkl is a simple classifier on a few numerical features:
        # Create a dummy feature vector. You MUST replace this with actual features
        # your ml_screening_model.pkl was trained on.
        # Example: if trained on [experience, normalized_similarity, skill_match_percentage]
        features_for_prediction = np.array([[experience if experience is not None else 0, 
                                             normalized_similarity, 
                                             skill_match_percentage]])
        
        try:
            prediction = ml_model.predict(features_for_prediction)[0]
            prediction_proba = ml_model.predict_proba(features_for_prediction)[0]
            # Assuming binary classification: 0 for Reject, 1 for Hire
            hire_probability = prediction_proba[1] * 100 if len(prediction_proba) > 1 else (100 if prediction == 1 else 0)
            screening_outcome = "Recommended for Interview" if prediction == 1 else "Not Recommended (Reject)"
        except Exception as e:
            st.warning(f"Could not make a screening prediction (ML model error): {e}. Displaying only extracted info and similarity.")
            screening_outcome = "Prediction Unavailable"
            hire_probability = "N/A"

    # --- Generate Certificate Link (THIS IS THE CERTIFICATION PART) ---
    # The CERTIFICATE_BASE_URL is passed from main.py via session state.
    CERTIFICATE_BASE_URL = st.session_state.get('CERTIFICATE_BASE_URL', 'http://localhost:8501/Certificate_Page') # Fallback

    certificate_id = str(uuid.uuid4()) # Generate a unique ID for the certificate
    # Encode parameters for URL
    params = {
        "id": certificate_id,
        "name": name if name else "Candidate",
        "score": f"{normalized_similarity:.1f}",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "jd_used": urllib.parse.quote(st.session_state.get('jd_file_name', 'General Job Description'))
    }
    # Construct the certificate URL
    certificate_link = f"{CERTIFICATE_BASE_URL}?{urllib.parse.urlencode(params)}"

    # --- Prepare Results ---
    results = {
        "Resume Text": resume_text,
        "Extracted Information": {
            "Name": name,
            "Email": email,
            "Phone Number": phone,
            "Location": location,
            "Years of Experience": experience,
            "CGPA (Normalized to 4.0)": cgpa,
            "Education": education_details,
            "Work History": work_history_details,
            "Projects": project_details,
            "Skills Found": list(raw_resume_skills),
            "Categorized Skills": categorized_resume_skills,
            "Languages": extracted_languages
        },
        "Similarity Score (Resume vs. JD)": f"{normalized_similarity:.2f}%",
        "Skill Match Percentage": f"{skill_match_percentage:.2f}%",
        "Screening Outcome": screening_outcome,
        "Probability of Hire": f"{hire_probability:.2f}%" if isinstance(hire_probability, float) else hire_probability,
        "Certificate Link": certificate_link,
        "Certificate Data (for Firestore)": {
            "certificate_id": certificate_id,
            "name": name if name else "Candidate",
            "score": normalized_similarity,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "jd_used": st.session_state.get('jd_file_name', 'General Job Description'),
            "timestamp": datetime.now().isoformat()
        }
    }
    return results

# --- Visualization Functions ---
def plot_skill_wordcloud(skills_dict, title="Skills Word Cloud"):
    """Generates and displays a word cloud from categorized skills."""
    all_skills = [skill for category, skills in skills_dict.items() for skill in skills]
    if not all_skills:
        st.warning("No skills found to generate word cloud.")
        return

    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(" ".join(all_skills))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig) # Close the figure to prevent display issues

def plot_skill_category_breakdown(categorized_skills_dict):
    """Plots a bar chart of skill categories."""
    if not categorized_skills_dict:
        st.warning("No categorized skills to plot.")
        return

    category_counts = {category: len(skills) for category, skills in categorized_skills_dict.items()}
    df_categories = pd.DataFrame(category_counts.items(), columns=['Category', 'Count']).sort_values(by='Count', ascending=False)

    if df_categories.empty:
        st.warning("No categorized skills to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Count', y='Category', data=df_categories, palette='viridis', ax=ax)
    ax.set_title('Skill Category Breakdown')
    ax.set_xlabel('Number of Skills')
    ax.set_ylabel('Skill Category')
    st.pyplot(fig)
    plt.close(fig) # Close the figure to prevent display issues

def plot_score_gauge(score, title="Match Score"):
    """
    Creates a simple gauge chart for the match score.
    This is a simplified visual representation.
    """
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    
    # Define the gauge colors and ranges
    colors = ['#FF4B4B', '#FFA500', '#ADD8E6', '#90EE90'] # Red, Orange, Light Blue, Light Green
    ranges = [0, 25, 50, 75, 100]
    
    # Draw the arcs
    for i in range(len(colors)):
        ax.bar(x=np.radians(90), width=np.radians(45), height=1, bottom=ranges[i]/100, 
               color=colors[i], align='edge', edgecolor='white', linewidth=2, alpha=0.7)

    # Convert score to radians for placement on the arc
    angle = np.radians((score / 100) * 180) # Map 0-100 to 0-180 degrees
    
    # Draw the needle
    ax.plot([0, angle], [0, 1], color='black', linewidth=3, linestyle='-', marker='>', markersize=10)
    
    # Set limits and remove unnecessary elements
    ax.set_theta_zero_location("W") # Start from West (left)
    ax.set_theta_direction(-1) # Go clockwise
    ax.set_rticks([]) # Remove radial ticks
    ax.set_xticks(np.radians(np.linspace(0, 180, 5))) # Set angular ticks for 0, 45, 90, 135, 180
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_rlim(0, 1) # Set radial limits
    ax.set_title(f"{title}: {score:.2f}%", va='bottom', fontsize=16)
    
    st.pyplot(fig)
    plt.close(fig) # Close the figure to prevent display issues


# --- Streamlit UI ---
def main(): # This is the 'main' function that main.py expects to import
    st.set_page_config(layout="wide", page_title="ScreenerPro - AI Resume Screener")

    st.title("📄 ScreenerPro: AI-Powered Resume Screener")
    st.markdown("""
    Upload a candidate's resume (PDF) and provide a job description.
    Our AI will extract key information, calculate a match score, and provide insights.
    """)

    # Job Description Input
    st.header("1. Provide Job Description")
    jd_upload_option = st.radio("How would you like to provide the Job Description?", ("Upload JD File (PDF)", "Paste JD Text"))

    job_description_text = ""
    jd_file_name = "Manual JD Input"

    if jd_upload_option == "Upload JD File (PDF)":
        jd_file = st.file_uploader("Upload Job Description PDF", type=["pdf"], key="jd_file_uploader")
        if jd_file:
            st.info("Extracting text from Job Description PDF...")
            job_description_text = extract_text_from_pdf(jd_file)
            jd_file_name = jd_file.name
            if "[ERROR]" in job_description_text:
                st.error(f"Error extracting text from JD: {job_description_text}")
                job_description_text = "" # Clear invalid text
            else:
                st.success("Job Description PDF processed successfully!")
                with st.expander("View Extracted Job Description Text"):
                    st.text_area("Extracted JD Text", job_description_text, height=200, disabled=True)
    else:
        job_description_text = st.text_area("Paste Job Description Text Here", height=300, 
                                            placeholder="e.g., 'We are looking for a Data Scientist with strong Python skills...'")
        jd_file_name = "Pasted JD Text"
    
    # Store JD file name in session state for certificate
    st.session_state['jd_file_name'] = jd_file_name

    st.header("2. Upload Candidate Resume")
    uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"], key="resume_file_uploader")

    process_button = st.button("✨ Process Resume")

    if process_button:
        if not job_description_text:
            st.error("Please provide a Job Description before processing the resume.")
        elif not uploaded_file:
            st.error("Please upload a Resume PDF to process.")
        else:
            with st.spinner("Processing resume and generating insights... This may take a moment, especially for scanned PDFs."):
                # Reset file pointer for the resume file before processing
                uploaded_file.seek(0)
                results = process_resume(uploaded_file, job_description_text)

            if results:
                st.success("Resume processed successfully!")
                
                st.subheader("📊 Screening Results")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Overall Match Score", results["Similarity Score (Resume vs. JD)"])
                    st.metric("Skill Match Percentage", results["Skill Match Percentage"])
                    st.metric("AI Screening Outcome", results["Screening Outcome"])
                    if isinstance(results["Probability of Hire"], str):
                        st.metric("Probability of Hire", results["Probability of Hire"])
                    else:
                        st.metric("Probability of Hire", f"{results['Probability of Hire']:.2f}%")

                    # Display the certificate link prominently
                    st.markdown(f"### 🎉 Official Certificate:")
                    st.markdown(f"Click [here]({results['Certificate Link']}) to view/download the candidate's certificate.", unsafe_allow_html=True)
                    st.info("Share this link with the candidate as proof of their assessment!")
                    
                    # Store certificate data in session state to be picked up by main.py for Firestore
                    st.session_state['certificate_data_to_save'] = results['Certificate Data (for Firestore)']

                with col2:
                    st.write("### Match Score Gauge")
                    # Convert score to float for the gauge plot
                    score_value = float(results["Similarity Score (Resume vs. JD)"].replace('%', ''))
                    plot_score_gauge(score_value)

                st.subheader("🔍 Detailed Resume Insights")

                st.markdown("#### Extracted Personal Information")
                personal_info_df = pd.DataFrame([
                    {"Field": "Name", "Value": results["Extracted Information"]["Name"]},
                    {"Field": "Email", "Value": results["Extracted Information"]["Email"]},
                    {"Field": "Phone Number", "Value": results["Extracted Information"]["Phone Number"]},
                    {"Field": "Location", "Value": results["Extracted Information"]["Location"]},
                    {"Field": "Years of Experience", "Value": results["Extracted Information"]["Years of Experience"]},
                    {"Field": "CGPA (Normalized to 4.0)", "Value": results["Extracted Information"]["CGPA (Normalized to 4.0)"]},
                    {"Field": "Languages", "Value": results["Extracted Information"]["Languages"]}
                ])
                st.table(personal_info_df)

                st.markdown("#### Education History")
                if results["Extracted Information"]["Education"]:
                    st.dataframe(pd.DataFrame(results["Extracted Information"]["Education"]))
                else:
                    st.info("No detailed education history extracted.")

                st.markdown("#### Work History")
                if results["Extracted Information"]["Work History"]:
                    st.dataframe(pd.DataFrame(results["Extracted Information"]["Work History"]))
                else:
                    st.info("No detailed work history extracted.")

                st.markdown("#### Projects")
                if results["Extracted Information"]["Projects"]:
                    st.dataframe(pd.DataFrame(results["Extracted Information"]["Projects"]))
                else:
                    st.info("No detailed project information extracted.")

                st.markdown("#### Skills Analysis")
                st.write(f"**Skills found in Resume:** {', '.join(results['Extracted Information']['Skills Found'])}")
                st.write(f"**Skills required by Job Description:** {', '.join(jd_skills)}") # Corrected to show JD skills
                st.write(f"**Matching Skills:** {', '.join(results['Extracted Information']['Skills Found'].intersection(jd_skills))}") # Corrected to show matching skills

                st.markdown("---")
                st.subheader("Visualizations")
                
                col_viz1, col_viz2 = st.columns(2)
                with col_viz1:
                    plot_skill_wordcloud(results["Extracted Information"]["Categorized Skills"], "Resume Skills Word Cloud")
                with col_viz2:
                    plot_skill_category_breakdown(results["Extracted Information"]["Categorized Skills"])

                with st.expander("View Raw Extracted Resume Text"):
                    st.text_area("Resume Text", results["Resume Text"], height=300, disabled=True)

            else:
                st.error("Failed to process resume. Please check the uploaded file and job description.")

if __name__ == "__main__":
    main()

