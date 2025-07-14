# Version 1.18 - Enhanced Table UI, More Filters, and Shortlist Feature
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

# Set Streamlit page configuration for wide layout
st.set_page_config(layout="wide")

# Download NLTK stopwords data if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- Load Embedding + ML Model ---
@st.cache_resource
def load_ml_model():
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        ml_model = joblib.load("ml_screening_model.pkl")
        return model, ml_model
    except Exception as e:
        st.error(f"❌ Error loading ML models: {e}. Please ensure 'ml_screening_model.pkl' is in the same directory.")
        return None, None

model, ml_model = load_ml_model()

# --- Predefined List of Cities for Location Extraction ---
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


# --- Stop Words List (Using NLTK) ---
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

# --- Skill Categories (for categorization and weighting) ---
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
MASTER_SKILLS = set([skill for category_list in SKILL_CATEGORIES.values() for skill in category_list])


# --- Helpers ---
def clean_text(text):
    """Cleans text by removing newlines, extra spaces, and non-ASCII characters."""
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
                extracted_keywords.add(skill_phrase.lower()) # Add the original skill (lowercase)
                # Categorize the skill
                found_category = False
                for category, skills_in_category in SKILL_CATEGORIES.items():
                    if skill_phrase.lower() in [s.lower() for s in skills_in_category]:
                        categorized_keywords[category].append(skill_phrase)
                        found_category = True
                        break
                if not found_category:
                    categorized_keywords["Uncategorized"].append(skill_phrase) # Add to uncategorized if no match

                # Replace the found skill with placeholders to avoid re-matching parts of it
                temp_text = re.sub(pattern, " ", temp_text)
        
        # After extracting phrases, now extract individual words that are in the filter_set
        # and haven't been part of a multi-word skill already extracted.
        # This ensures single-word skills from MASTER_SKILLS are also captured.
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

    else: # Fallback: if no specific filter_set (MASTER_SKILLS is empty), use the default STOP_WORDS logic
        all_words = set(re.findall(r'\b\w+\b', cleaned_text))
        extracted_keywords = {word for word in all_words if word not in STOP_WORDS}
        for word in extracted_keywords:
            categorized_keywords["General Keywords"].append(word) # Default category for fallback

    return extracted_keywords, dict(categorized_keywords)


def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            return ''.join(page.extract_text() or '' for page in pdf.pages)
    except Exception as e:
        return f"[ERROR] {str(e)}"

def extract_years_of_experience(text):
    """Extracts years of experience from a given text by parsing date ranges or keywords."""
    text = text.lower()
    total_months = 0
    job_date_ranges = re.findall(
        r'(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})\s*(?:to|–|-)\s*(present|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})',
        text
    )

    for start, end in job_date_ranges:
        try:
            start_date = datetime.strptime(start.strip(), '%b %Y')
        except ValueError:
            try:
                start_date = datetime.strptime(start.strip(), '%B %Y')
            except ValueError:
                continue

        if end.strip() == 'present':
            end_date = datetime.now()
        else:
            try:
                end_date = datetime.strptime(end.strip(), '%b %Y')
            except ValueError:
                try:
                    end_date = datetime.strptime(end.strip(), '%B %Y')
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
    """Extracts an email address from the given text."""
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else None

def extract_phone_number(text):
    """Extracts a phone number from the given text."""
    # Common patterns: (XXX) XXX-XXXX, XXX-XXX-XXXX, XXXXXXXXXX, XXX.XXX.XXXX
    # This regex is more robust for various formats
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
    EXCLUDE_NAME_TERMS = {"linkedin", "github", "portfolio", "resume", "cv", "profile", "contact", "email", "phone"}

    potential_name_lines = []
    # Consider the first 5 lines for name extraction
    for line in lines[:5]:
        line = line.strip()
        line_lower = line.lower()

        # Filter out lines that clearly contain email, phone, or too many words
        # Also, filter out lines that contain any of the EXCLUDE_NAME_TERMS
        if not re.search(r'[@\d\.\-]', line) and \
           len(line.split()) <= 4 and \
           not any(term in line_lower for term in EXCLUDE_NAME_TERMS):
            # Heuristic: Check if the line is mostly capitalized words (like a name)
            if line.isupper() or (line and line[0].isupper() and all(word[0].isupper() or not word.isalpha() for word in line.split())):
                potential_name_lines.append(line)

    if potential_name_lines:
        # Prioritize longer potential names, then filter out common resume section headers
        name = max(potential_name_lines, key=len)
        name = re.sub(r'summary|education|experience|skills|projects|certifications|profile|contact', '', name, flags=re.IGNORECASE).strip()
        # Further clean up any leading/trailing non-alphabetic characters if they remain
        name = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', name).strip()
        if name:
            return name.title()
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
                        technologies.append(skill)
                
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
    Looks for a "Languages" section and lists known languages.
    """
    languages_list = []
    text_lower = text.lower()

    # Define a comprehensive list of languages (Indian and Foreign)
    all_languages = [
        "english", "hindi", "spanish", "french", "german", "mandarin", "japanese", "arabic",
        "russian", "portuguese", "italian", "korean", "bengali", "marathi", "telugu", "tamil",
        "gujarati", "urdu", "kannada", "odia", "malayalam", "punjabi", "assamese", "kashmiri",
        "sindhi", "sanskrit", "dutch", "swedish", "norwegian", "danish", "finnish", "greek",
        "turkish", "hebrew", "thai", "vietnamese", "indonesian", "malay", "filipino", "swahili",
        "farsi", "persian", "polish", "ukrainian", "romanian", "czech", "slovak", "hungarian"
    ]
    
    # Look for a "Languages" section header
    languages_section_match = re.search(r'\b(languages|language skills|linguistic abilities)\b\s*(\n|$)', text_lower)
    
    if languages_section_match:
        start_index = languages_section_match.end()
        # Define potential end markers for the languages section
        sections = ['education', 'experience', 'work history', 'skills', 'projects', 'certifications', 'awards', 'publications', 'interests', 'hobbies']
        end_index = len(text_lower)
        for section in sections:
            section_match = re.search(r'\b' + re.escape(section) + r'\b', text_lower[start_index:], re.IGNORECASE)
            if section_match:
                end_index = start_index + section_match.start()
                break
        
        languages_text = text_lower[start_index:end_index].strip()
        
        # Extract languages from the identified section
        for lang in all_languages:
            if re.search(r'\b' + re.escape(lang) + r'\b', languages_text):
                languages_list.append(lang.title())
    else:
        # Fallback: if no explicit section, try to find languages anywhere in the text
        # This is less precise but can catch languages mentioned in summary/profile
        for lang in all_languages:
            if re.search(r'\b' + re.escape(lang) + r'\b', text_lower):
                if lang.title() not in languages_list: # Avoid duplicates if found in multiple places
                    languages_list.append(lang.title())

    return ", ".join(sorted(list(set(languages_list)))) if languages_list else "Not Found"


def format_education_details(edu_list):
    """Formats a list of education dictionaries into a readable string."""
    if not edu_list:
        return "Not Found"
    formatted_entries = []
    for entry in edu_list:
        parts = []
        if entry.get("Degree"):
            parts.append(entry["Degree"])
        if entry.get("Major"):
            parts.append(f"in {entry['Major']}")
        if entry.get("University"):
            parts.append(f"from {entry['University']}")
        if entry.get("Year"):
            parts.append(f"({entry['Year']})")
        formatted_entries.append(" ".join(parts).strip())
    return "; ".join(formatted_entries) if formatted_entries else "Not Found"

def format_work_history(work_list):
    """Formats a list of work history dictionaries into a readable string."""
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
    """Formats a list of project dictionaries into a readable string."""
    if not proj_list:
        return "Not Found"
    formatted_entries = []
    for entry in proj_list:
        parts = []
        if entry.get("Project Title"):
            parts.append(f"**{entry['Project Title']}**")
        if entry.get("Technologies Used"):
            parts.append(f"({entry['Technologies Used']})")
        # Description can be long, so maybe just a snippet or indicate presence
        if entry.get("Description") and entry["Description"].strip():
            desc_snippet = entry["Description"].split('\n')[0][:50] + "..." if len(entry["Description"]) > 50 else entry["Description"]
            parts.append(f'"{desc_snippet}"')
        formatted_entries.append(" ".join(parts).strip())
    return "; ".join(formatted_entries) if formatted_entries else "Not Found"


# --- Concise AI Suggestion Function (for table display) ---
@st.cache_data(show_spinner="Generating concise AI Suggestion...")
def generate_concise_ai_suggestion(candidate_name, score, years_exp, semantic_similarity, cgpa):
    """
    Generates a concise AI suggestion based on rules, focusing on overall fit and key points.
    Now includes CGPA in the assessment.
    """
    overall_fit_description = ""
    review_focus_text = ""
    key_strength_hint = ""

    # Define thresholds
    high_score = 85
    moderate_score = 65
    high_exp = 4
    moderate_exp = 2
    high_sem_sim = 0.75
    moderate_sem_sim = 0.4
    high_cgpa = 3.5 # Assuming normalized to 4.0 scale
    moderate_cgpa = 3.0

    # Base assessment
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

    # Incorporate CGPA into the suggestion
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

# --- Detailed HR Assessment Function (for top candidate display) ---
@st.cache_data(show_spinner="Generating detailed HR Assessment...")
def generate_detailed_hr_assessment(candidate_name, score, years_exp, semantic_similarity, cgpa, jd_text, resume_text, matched_keywords, missing_skills, max_exp_cutoff):
    """
    Generates a detailed, multi-paragraph HR assessment for a candidate.
    Now includes matched and missing skills, CGPA, and considers max experience.
    """
    assessment_parts = []
    overall_assessment_title = ""
    next_steps_focus = ""

    # Convert lists to strings for display if they are lists
    matched_kws_str = ", ".join(matched_keywords) if isinstance(matched_keywords, list) else matched_keywords
    missing_skills_str = ", ".join(missing_skills) if isinstance(missing_skills, list) else missing_skills

    # Define thresholds
    high_score = 90
    strong_score = 80
    promising_score = 60
    high_exp = 5
    strong_exp = 3
    promising_exp = 1
    high_sem_sim = 0.85
    strong_sem_sim = 0.7
    promising_sem_sim = 0.35
    high_cgpa = 3.5 # Assuming normalized to 4.0 scale
    strong_cgpa = 3.0
    promising_cgpa = 2.5

    # Tier 1: Exceptional Candidate
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

    # Tier 2: Strong Candidate
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

    # Tier 3: Promising Candidate
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

    # Tier 4: Limited Match
    else:
        overall_assessment_title = "Limited Match: Consider Only for Niche Needs or Pipeline Building"
        assessment_parts.append(f"**{candidate_name}** shows a **limited match** with a score of {score:.2f}% and {years_exp:.1f} years of experience (semantic similarity: {semantic_similarity:.2f}). This profile indicates a significant deviation from the core requirements of the job description.")
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


def semantic_score(resume_text, jd_text, years_exp, cgpa, high_priority_skills, medium_priority_skills, jd_skills_filter_set): # Added jd_skills_filter_set
    """
    Calculates a semantic score using an ML model and provides additional details.
    Falls back to smart_score if the ML model is not loaded or prediction fails.
    Applies STOP_WORDS filtering for keyword analysis (internally, not for display).
    Now includes CGPA and weighted keyword matching in the scoring.
    """
    jd_clean = clean_text(jd_text)
    resume_clean = clean_text(resume_text)

    score = 0.0
    semantic_similarity = 0.0

    # Extract raw skills for scoring
    resume_raw_skills, _ = extract_relevant_keywords(resume_clean, MASTER_SKILLS)
    jd_raw_skills, _ = extract_relevant_keywords(jd_clean, jd_skills_filter_set) # Use the passed filter set for JD

    # Calculate weighted keyword overlap
    weighted_keyword_overlap_score = 0
    total_jd_skill_weight = 0

    # Define weights
    WEIGHT_HIGH = 3
    WEIGHT_MEDIUM = 2
    WEIGHT_BASE = 1

    for jd_skill in jd_raw_skills:
        current_weight = WEIGHT_BASE
        if jd_skill in [s.lower() for s in high_priority_skills]:
            current_weight = WEIGHT_HIGH
        elif jd_skill in [s.lower() for s in medium_priority_skills]:
            current_weight = WEIGHT_MEDIUM
        
        total_jd_skill_weight += current_weight
        
        if jd_skill in resume_raw_skills:
            weighted_keyword_overlap_score += current_weight

    if total_jd_skill_weight > 0:
        weighted_jd_coverage_percentage = (weighted_keyword_overlap_score / total_jd_skill_weight) * 100
    else:
        weighted_jd_coverage_percentage = 0.0


    if ml_model is None or model is None:
        st.warning("ML models not loaded. Providing basic score and generic feedback.")
        # Simplified fallback for score and feedback
        basic_score = (weighted_jd_coverage_percentage * 0.7) # Use weighted coverage
        basic_score += min(years_exp * 5, 30) # Add up to 30 for experience
        
        # Add a small CGPA bonus/penalty for fallback score
        if cgpa is not None:
            if cgpa >= 3.5:
                basic_score += 5
            elif cgpa < 2.5:
                basic_score -= 5
        
        score = round(min(basic_score, 100), 2)
        
        return score, round(semantic_similarity, 2)

    try:
        jd_embed = model.encode(jd_clean)
        resume_embed = model.encode(resume_clean)

        semantic_similarity = cosine_similarity(jd_embed.reshape(1, -1), resume_embed.reshape(1, -1))[0][0]
        semantic_similarity = float(np.clip(semantic_similarity, 0, 1))

        years_exp_for_model = float(years_exp) if years_exp is not None else 0.0

        # Features for the ML model - now using weighted_keyword_overlap_score
        features = np.concatenate([jd_embed, resume_embed, [years_exp_for_model], [weighted_keyword_overlap_score]])

        predicted_score = ml_model.predict([features])[0]

        blended_score = (predicted_score * 0.6) + \
                        (weighted_jd_coverage_percentage * 0.1) + \
                        (semantic_similarity * 100 * 0.3)

        if semantic_similarity > 0.7 and years_exp >= 3:
            blended_score += 5
        
        # Adjust score based on CGPA (if available)
        if cgpa is not None:
            if cgpa >= 3.5: # Excellent CGPA
                blended_score += 3
            elif cgpa >= 3.0: # Good CGPA
                blended_score += 1
            elif cgpa < 2.5: # Lower CGPA, slight penalty
                blended_score -= 2


        score = float(np.clip(blended_score, 0, 100))
        
        return round(score, 2), round(semantic_similarity, 2)

    except Exception as e:
        st.warning(f"Error during semantic scoring, falling back to basic: {e}")
        # Simplified fallback for score and feedback if ML prediction fails
        basic_score = (weighted_jd_coverage_percentage * 0.7) # Use weighted coverage
        basic_score += min(years_exp * 5, 30) # Add up to 30 for experience
        
        # Add a small CGPA bonus/penalty for fallback score
        if cgpa is not None:
            if cgpa >= 3.5:
                basic_score += 5
            elif cgpa < 2.5:
                basic_score -= 5

        score = round(min(basic_score, 100), 2)

        return score, 0.0 # Return 0 for semantic similarity on fallback


# --- Email Generation Function ---
def create_mailto_link(recipient_email, candidate_name, job_title="Job Opportunity", sender_name="Recruiting Team"):
    """
    Generates a mailto: link with pre-filled subject and body for inviting a candidate.
    """
    subject = urllib.parse.quote(f"Invitation for Interview - {job_title} - {candidate_name}")
    body = urllib.parse.quote(f"""Dear {candidate_name},

We were very impressed with your profile and would like to invite you for an interview for the {job_title} position.

Best regards,

The {sender_name}""")
    return f"mailto:{recipient_email}?subject={subject}&body={body}"

# --- Function to encapsulate the Resume Screener logic ---
def resume_screener_page():
    st.title("🧠 ScreenerPro – AI-Powered Resume Screener")

    # --- Job Description and Controls Section ---
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
        
        # Determine the JD name to be stored in results
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
        # Store cutoff and min_experience in session state
        cutoff = st.slider("📈 **Minimum Score Cutoff (%)**", 0, 100, 75, help="Candidates scoring below this percentage will be flagged for closer review or considered less suitable.")
        st.session_state['screening_cutoff_score'] = cutoff # Store in session state

        min_experience = st.slider("💼 **Minimum Experience Required (Years)**", 0, 15, 2, help="Candidates with less than this experience will be noted.")
        st.session_state['screening_min_experience'] = min_experience # Store in session state

        max_experience = st.slider("⬆️ **Maximum Experience Allowed (Years)**", 0, 20, 10, help="Candidates with more than this experience might be considered overqualified or outside the target range.")
        st.session_state['screening_max_experience'] = max_experience # Store in session state

        # New CGPA Cutoff Slider
        min_cgpa = st.slider("🎓 **Minimum CGPA Required (4.0 Scale)**", 0.0, 4.0, 2.5, 0.1, help="Candidates with CGPA below this value (normalized to 4.0) will be noted.")
        st.session_state['screening_min_cgpa'] = min_cgpa # Store in session state

        st.markdown("---")
        st.info("Once criteria are set, upload resumes below to begin screening.")

    # --- Skill Weighting Section ---
    st.markdown("## 🎯 Skill Prioritization (Optional)")
    st.caption("Assign higher importance to specific skills in the Job Description.")
    
    # Use the dynamically generated MASTER_SKILLS for selection
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
            options=[s for s in all_master_skills if s not in high_priority_skills], # Prevent overlap
            help="Select skills that are very important, but not as critical as high priority ones."
        )

    # Custom Skill Keywords input
    custom_jd_skills_input = st.text_input(
        "➕ **Add Custom JD Skills (comma-separated)**",
        help="Enter additional skills relevant to the Job Description, e.g., 'Leadership, Cloud Security, Microservices Architecture'. These will be considered in scoring."
    )
    custom_jd_skills_set = {s.strip().lower() for s in custom_jd_skills_input.split(',') if s.strip()}
    
    # Combine MASTER_SKILLS with custom JD skills for JD processing
    jd_skills_for_processing = MASTER_SKILLS.union(custom_jd_skills_set)

    resume_files = st.file_uploader("📄 **Upload Resumes (PDF)**", type="pdf", accept_multiple_files=True, help="Upload one or more PDF resumes for screening.")

    df = pd.DataFrame()

    if jd_text and resume_files:
        # --- Job Description Keyword Cloud ---
        st.markdown("---")
        st.markdown("## ☁️ Job Description Keyword Cloud")
        st.caption("Visualizing the most frequent and important keywords from the Job Description.")
        st.info("💡 To filter candidates by these skills, use the 'Filter Candidates by Skill' section below the main results table.") # New instruction
        
        # Use all_master_skills for cloud generation
        jd_words_for_cloud_set, _ = extract_relevant_keywords(jd_text, jd_skills_for_processing) # Use combined skills for JD cloud
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
        resume_text_map = {}
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
            location = extract_location(text) # Now uses custom city list
            languages = extract_languages(text)
            
            # Extract structured details
            education_details_raw = extract_education_details(text)
            work_history_raw = extract_work_history(text)
            project_details_raw = extract_project_details(text)

            # Format structured details for display in the DataFrame
            education_details_formatted = format_education_details(education_details_raw)
            work_history_formatted = format_work_history(work_history_raw)
            project_details_formatted = format_project_details(project_details_raw)

            candidate_name = extract_name(text) or file.name.replace('.pdf', '').replace('_', ' ').title()
            cgpa = extract_cgpa(text)

            # Calculate Matched Keywords and Missing Skills using the new function
            resume_raw_skills_set, resume_categorized_skills = extract_relevant_keywords(text, all_master_skills) # Resume skills still against master
            jd_raw_skills_set, jd_categorized_skills = extract_relevant_keywords(jd_text, jd_skills_for_processing) # JD skills against combined set

            matched_keywords = list(resume_raw_skills_set.intersection(jd_raw_skills_set))
            missing_skills = list(jd_raw_skills_set.difference(resume_raw_skills_set)) 

            score, semantic_similarity = semantic_score(text, jd_text, exp, cgpa, high_priority_skills, medium_priority_skills, jd_skills_for_processing) # Pass to semantic_score
            
            # Generate the CONCISE AI suggestion for the table
            concise_ai_suggestion = generate_concise_ai_suggestion(
                candidate_name=candidate_name,
                score=score,
                years_exp=exp,
                semantic_similarity=semantic_similarity,
                cgpa=cgpa
            )

            # Generate the DETAILED HR assessment for the top candidate
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
                "JD Used": jd_name_for_results
            })
            resume_text_map[file.name] = text
        
        progress_bar.empty()
        status_text.empty()


        df = pd.DataFrame(results).sort_values(by="Score (%)", ascending=False).reset_index(drop=True)

        # Add 'Shortlisted' and 'Notes' columns if they don't exist
        if 'Shortlisted' not in df.columns:
            df['Shortlisted'] = False
        if 'Notes' not in df.columns:
            df['Notes'] = ""

        # Initialize or update the full_results_df in session state
        if 'full_results_df' not in st.session_state:
            st.session_state['full_results_df'] = df
        else:
            # Merge new results with existing state, preserving user edits for 'Shortlisted' and 'Notes'
            existing_df = st.session_state['full_results_df'].set_index('File Name')
            new_df_indexed = df.set_index('File Name')

            # Use combine_first to prioritize existing user edits
            # This will keep 'Shortlisted' and 'Notes' from existing_df if they exist for a given File Name
            # and fill other columns/new rows from new_df_indexed.
            combined_df = new_df_indexed.combine_first(existing_df)
            st.session_state['full_results_df'] = combined_df.reset_index()

        # --- Overall Candidate Comparison Chart ---
        st.markdown("## 📊 Candidate Score Comparison")
        st.caption("Visual overview of how each candidate ranks against the job requirements.")
        if not df.empty:
            fig, ax = plt.subplots(figsize=(12, 7))
            # Define colors: Green for top, Yellow for moderate, Red for low
            colors = ['#4CAF50' if s >= cutoff else '#FFC107' if s >= (cutoff * 0.75) else '#F44346' for s in df['Score (%)']]
            bars = ax.bar(df['Candidate Name'], df['Score (%)'], color=colors)
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

        # --- TOP CANDIDATE AI RECOMMENDATION (Game Changer Feature) ---
        st.markdown("## 👑 Top Candidate AI Assessment")
        st.caption("A concise, AI-powered assessment for the most suitable candidate.")
        
        if not df.empty:
            top_candidate = df.iloc[0] # Get the top candidate (already sorted by score)
            
            # Safely format CGPA and Semantic Similarity for display
            cgpa_display = f"{top_candidate['CGPA (4.0 Scale)']:.2f}" if top_candidate['CGPA (4.0 Scale)'] is not None else "N/A"
            semantic_sim_display = f"{top_candidate['Semantic Similarity']:.2f}" if top_candidate['Semantic Similarity'] is not None else "N/A"

            st.markdown(f"### **{top_candidate['Candidate Name']}**")
            st.markdown(f"**Score:** {top_candidate['Score (%)']:.2f}% | **Experience:** {top_candidate['Years Experience']:.1f} years | **CGPA:** {cgpa_display} (4.0 Scale) | **Semantic Similarity:** {semantic_sim_display}")
            st.markdown(f"**AI Assessment:**")
            st.markdown(top_candidate['Detailed HR Assessment']) # Display the detailed HR assessment here
            
            # Display Categorized Matched Skills for the top candidate
            st.markdown("#### Matched Skills Breakdown:")
            if top_candidate['Matched Keywords (Categorized)']:
                for category, skills in top_candidate['Matched Keywords (Categorized)'].items():
                    st.write(f"**{category}:** {', '.join(skills)}")
            else:
                st.write("No categorized matched skills found.")

            # Display Categorized Missing Skills for the top candidate
            st.markdown("#### Missing Skills Breakdown (from JD):")
            # Need to re-calculate missing skills based on JD's categorized skills and candidate's raw skills
            jd_raw_skills_set_for_top, jd_categorized_skills_for_top = extract_relevant_keywords(jd_text, jd_skills_for_processing) # Use combined skills
            resume_raw_skills_set_for_top, _ = extract_relevant_keywords(top_candidate['Resume Raw Text'], all_master_skills)
            
            missing_skills_for_top = jd_raw_skills_set_for_top.difference(resume_raw_skills_set_for_top)
            
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
                
                for category, skills in missing_categorized.items():
                    st.write(f"**{category}:** {', '.join(skills)}")
            else:
                st.write("No missing skills found for this candidate relative to the JD.")


            # Action button for the top candidate
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


        # === AI Recommendation for Shortlisted Candidates (Streamlined) ===
        # This section now focuses on a quick summary for *all* shortlisted,
        # with the top one highlighted above.
        st.markdown("## 🌟 AI Shortlisted Candidates Overview")
        st.caption("Candidates automatically shortlisted based on your score, experience, and CGPA criteria.")

        # Updated filtering for AI shortlisted candidates to include CGPA and max experience
        ai_shortlisted_candidates = st.session_state['full_results_df'][
            (st.session_state['full_results_df']['Score (%)'] >= cutoff) & 
            (st.session_state['full_results_df']['Years Experience'] >= min_experience) &
            (st.session_state['full_results_df']['Years Experience'] <= max_experience) & # New: Max experience filter
            ((st.session_state['full_results_df']['CGPA (4.0 Scale)'].isnull()) | (st.session_state['full_results_df']['CGPA (4.0 Scale)'] >= min_cgpa)) # Handle cases where CGPA is not found
        ]

        if not ai_shortlisted_candidates.empty:
            st.success(f"**{len(ai_shortlisted_candidates)}** candidate(s) meet your specified criteria (Score ≥ {cutoff}%, Experience {min_experience}-{max_experience} years, CGPA ≥ {min_cgpa} or N/A).")
            
            # Display a concise table for AI shortlisted candidates
            display_shortlisted_summary_cols = [
                'Candidate Name',
                'Score (%)',
                'Years Experience',
                'CGPA (4.0 Scale)', # Added CGPA
                'Semantic Similarity',
                'Email', # Include email here for quick reference
                'AI Suggestion' # This is the concise AI suggestion
            ]
            
            st.dataframe(
                ai_shortlisted_candidates[display_shortlisted_summary_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Score (%)": st.column_config.ProgressColumn(
                        "Score (%)",
                        help="Matching score against job requirements",
                        format="%f",
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
                    )
                }
            )
            st.info("For individual detailed AI assessments and action steps, please refer to the table below or the Analytics Dashboard.")

        else:
            st.warning(f"No candidates met the defined screening criteria (score cutoff, experience between {min_experience}-{max_experience} years, and minimum CGPA). You might consider adjusting the sliders or reviewing the uploaded resumes/JD.")

        st.markdown("---")

        # Add a 'Tag' column for quick categorization - UPDATED WITH CGPA and Max Exp
        st.session_state['full_results_df']['Tag'] = st.session_state['full_results_df'].apply(lambda row: 
            "👑 Exceptional Match" if row['Score (%)'] >= 90 and row['Years Experience'] >= 5 and row['Years Experience'] <= max_experience and row['Semantic Similarity'] >= 0.85 and (row['CGPA (4.0 Scale)'] is None or row['CGPA (4.0 Scale)'] >= 3.5) else (
            "🔥 Strong Candidate" if row['Score (%)'] >= 80 and row['Years Experience'] >= 3 and row['Years Experience'] <= max_experience and row['Semantic Similarity'] >= 0.7 and (row['CGPA (4.0 Scale)'] is None or row['CGPA (4.0 Scale)'] >= 3.0) else (
            "✨ Promising Fit" if row['Score (%)'] >= 60 and row['Years Experience'] >= 1 and row['Years Experience'] <= max_experience and (row['CGPA (4.0 Scale)'] is None or row['CGPA (4.0 Scale)'] >= 2.5) else (
            "⚠️ Needs Review" if row['Score (%)'] >= 40 else 
            "❌ Limited Match"))), axis=1)

        st.markdown("## 📋 Comprehensive Candidate Results Table")
        st.caption("Full details for all processed resumes. Use filters below and edit 'Shortlisted' and 'Notes' directly.")
        
        # --- Interactive Filters for Comprehensive Table ---
        st.markdown("### Filter Options")
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        filter_col4, filter_col5, filter_col6 = st.columns(3)

        with filter_col1:
            score_min, score_max = st.slider(
                "Score Range (%)",
                0, 100, (0, 100),
                key="score_filter_slider",
                help="Filter candidates by their overall score."
            )
        with filter_col2:
            exp_min, exp_max = st.slider(
                "Experience Range (Years)",
                0.0, float(st.session_state['full_results_df']['Years Experience'].max() if not st.session_state['full_results_df'].empty else 15.0),
                (0.0, float(st.session_state['full_results_df']['Years Experience'].max() if not st.session_state['full_results_df'].empty else 15.0)), 0.1,
                key="exp_filter_slider",
                help="Filter candidates by their years of experience."
            )
        with filter_col3:
            cgpa_min, cgpa_max = st.slider(
                "CGPA Range (4.0 Scale)",
                0.0, 4.0, (0.0, 4.0), 0.1,
                key="cgpa_filter_slider",
                help="Filter candidates by their CGPA (normalized to a 4.0 scale)."
            )
        
        with filter_col4:
            all_tags = sorted(st.session_state['full_results_df']['Tag'].unique()) if not st.session_state['full_results_df'].empty else []
            selected_tags = st.multiselect(
                "Filter by Tag",
                options=all_tags,
                default=all_tags,
                key="tag_filter_multiselect",
                help="Select one or more tags to filter candidates."
            )
        
        with filter_col5:
            all_locations = sorted(st.session_state['full_results_df']['Location'].unique()) if not st.session_state['full_results_df'].empty else []
            selected_locations = st.multiselect(
                "Filter by Location",
                options=all_locations,
                default=all_locations,
                key="location_filter_multiselect",
                help="Select one or more locations to filter candidates."
            )
        
        with filter_col6:
            # Shortlist filter for the main table
            shortlist_filter_options = ["Show All", "Show Shortlisted Only", "Show Non-Shortlisted Only"]
            selected_shortlist_filter = st.selectbox(
                "Filter by Shortlist Status",
                options=shortlist_filter_options,
                key="shortlist_filter_selectbox",
                help="Filter the table to show only shortlisted, non-shortlisted, or all candidates."
            )

        # Apply filters to st.session_state['full_results_df']
        filtered_df_for_editor = st.session_state['full_results_df'].copy()
        
        filtered_df_for_editor = filtered_df_for_editor[
            (filtered_df_for_editor['Score (%)'] >= score_min) & (filtered_df_for_editor['Score (%)'] <= score_max) &
            (filtered_df_for_editor['Years Experience'] >= exp_min) & (filtered_df_for_editor['Years Experience'] <= exp_max) &
            ((filtered_df_for_editor['CGPA (4.0 Scale)'].isnull()) | ((filtered_df_for_editor['CGPA (4.0 Scale)'] >= cgpa_min) & (filtered_df_for_editor['CGPA (4.0 Scale)'] <= cgpa_max)))
        ]
        
        if selected_tags:
            filtered_df_for_editor = filtered_df_for_editor[filtered_df_for_editor['Tag'].isin(selected_tags)]
        
        if selected_locations:
            # This needs to handle multiple locations in the 'Location' column
            filtered_df_for_editor = filtered_df_for_editor[filtered_df_for_editor['Location'].apply(lambda x: any(loc in x for loc in selected_locations))]

        if selected_shortlist_filter == "Show Shortlisted Only":
            filtered_df_for_editor = filtered_df_for_editor[filtered_df_for_editor['Shortlisted'] == True]
        elif selected_shortlist_filter == "Show Non-Shortlisted Only":
            filtered_df_for_editor = filtered_df_for_editor[filtered_df_for_editor['Shortlisted'] == False]


        # Skill filter (from previous version, kept)
        jd_raw_skills_set, _ = extract_relevant_keywords(jd_text, jd_skills_for_processing) # Use combined skills
        all_unique_jd_skills = sorted(list(jd_raw_skills_set))
        selected_filter_skills = st.multiselect(
            "Filter by Specific Skills (from JD)",
            options=all_unique_jd_skills,
            key="skill_filter_multiselect",
            help="Only candidates possessing ALL selected skills will be shown."
        )
        if selected_filter_skills:
            for skill in selected_filter_skills:
                filtered_df_for_editor = filtered_df_for_editor[filtered_df_for_editor['Matched Keywords'].str.contains(r'\b' + re.escape(skill) + r'\b', case=False, na=False)]


        # Reset Filters button
        if st.button("Reset All Filters", key="reset_filters_button"):
            # Reset all filter-related session state variables
            # This will trigger a rerun and reset the UI elements to their defaults
            st.session_state['score_filter_slider'] = (0, 100)
            st.session_state['exp_filter_slider'] = (0.0, float(st.session_state['full_results_df']['Years Experience'].max() if not st.session_state['full_results_df'].empty else 15.0))
            st.session_state['cgpa_filter_slider'] = (0.0, 4.0)
            st.session_state['tag_filter_multiselect'] = all_tags
            st.session_state['location_filter_multiselect'] = all_locations
            st.session_state['shortlist_filter_selectbox'] = "Show All"
            st.session_state['skill_filter_multiselect'] = []
            st.rerun()


        # Define columns to display in the comprehensive table (reordered for user-friendliness)
        comprehensive_cols = [
            'Candidate Name',
            'Score (%)',
            'Tag',
            'AI Suggestion',
            'Years Experience',
            'CGPA (4.0 Scale)',
            'Semantic Similarity',
            'Email',
            'Phone Number',
            'Location',
            'Languages',
            'Matched Keywords',
            'Missing Skills',
            'Education Details',
            'Work History',
            'Project Details',
            'JD Used'
        ]
        
        final_display_cols = [col for col in comprehensive_cols if col in filtered_df_for_editor.columns]

        # Use st.data_editor for the main table to allow inline editing of 'Shortlisted' and 'Notes'
        edited_df = st.data_editor(
            filtered_df_for_editor[final_display_cols + ['Shortlisted', 'Notes']], # Add editable columns
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score (%)": st.column_config.ProgressColumn(
                    "Score (%)",
                    help="Matching score against job requirements",
                    format="%f",
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
                    help="AI's concise overall assessment and recommendation",
                    width="medium"
                ),
                "Tag": st.column_config.Column(
                    "Tag",
                    help="Quick categorization of candidate fit",
                    width="small"
                ),
                "Matched Keywords": st.column_config.Column(
                    "Matched Keywords",
                    help="Keywords found in both JD and Resume",
                    width="large"
                ),
                "Missing Skills": st.column_config.Column(
                    "Missing Skills",
                    help="Key skills from JD not found in Resume",
                    width="large"
                ),
                "JD Used": st.column_config.Column(
                    "JD Used",
                    help="Job Description used for this screening"
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
                    help="Structured education history (University, Degree, Major, Year)",
                    width="large"
                ),
                "Work History": st.column_config.Column(
                    "Work History",
                    help="Structured work experience (Company, Title, Dates)",
                    width="large"
                ),
                "Project Details": st.column_config.Column(
                    "Project Details",
                    help="Structured project experience (Title, Description, Technologies)",
                    width="large"
                ),
                "Shortlisted": st.column_config.CheckboxColumn( # Editable checkbox
                    "Shortlist",
                    help="Manually mark candidates for shortlisting",
                    default=False,
                ),
                "Notes": st.column_config.TextColumn( # Editable text area
                    "Notes",
                    help="Add personal notes for this candidate",
                    default="",
                    max_chars=200,
                )
            },
            key="comprehensive_table_editor" # Unique key for the data editor
        )

        # Update the session state DataFrame with the edited data from st.data_editor
        # This is crucial to persist changes made in the data editor
        # We need to merge edited_df back into the original full_results_df in session_state
        # to ensure that changes to 'Shortlisted' and 'Notes' persist even if filters change.
        if not edited_df.empty:
            # Create a temporary DataFrame from the edited_df with only 'File Name', 'Shortlisted', 'Notes'
            # to merge back into the main session state DataFrame.
            edited_subset = edited_df[['File Name', 'Shortlisted', 'Notes']].set_index('File Name')
            
            # Update the corresponding rows in the full_results_df
            current_full_df = st.session_state['full_results_df'].set_index('File Name')
            current_full_df.update(edited_subset) # Update existing rows
            st.session_state['full_results_df'] = current_full_df.reset_index()


        # --- CSV Download Button for Filtered Results ---
        csv_data = filtered_df_for_editor.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Filtered Results as CSV",
            data=csv_data,
            file_name="resume_screening_filtered_results.csv",
            mime="text/csv",
            help="Download the currently displayed table data as a CSV file."
        )

        # --- Shortlisted Candidates Display ---
        st.markdown("---")
        st.markdown("## ⭐ My Manually Shortlisted Candidates")
        st.caption("Candidates you have marked as 'Shortlisted' in the table above.")

        # Filter from the updated session state DataFrame for manually shortlisted candidates
        manually_shortlisted_df = st.session_state['full_results_df'][st.session_state['full_results_df']['Shortlisted'] == True]

        if not manually_shortlisted_df.empty:
            st.dataframe(
                manually_shortlisted_df[final_display_cols + ['Notes']], # Display same columns as main table + Notes
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Score (%)": st.column_config.ProgressColumn(
                        "Score (%)",
                        help="Matching score against job requirements",
                        format="%f",
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
                        help="AI's concise overall assessment and recommendation",
                        width="medium"
                    ),
                    "Tag": st.column_config.Column(
                        "Tag",
                        help="Quick categorization of candidate fit",
                        width="small"
                    ),
                    "Matched Keywords": st.column_config.Column(
                        "Matched Keywords",
                        help="Keywords found in both JD and Resume",
                        width="large"
                    ),
                    "Missing Skills": st.column_config.Column(
                        "Missing Skills",
                        help="Key skills from JD not found in Resume",
                        width="large"
                    ),
                    "JD Used": st.column_config.Column(
                        "JD Used",
                        help="Job Description used for this screening"
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
                        help="Structured education history (University, Degree, Major, Year)",
                        width="large"
                    ),
                    "Work History": st.column_config.Column(
                        "Work History",
                        help="Structured work experience (Company, Title, Dates)",
                        width="large"
                    ),
                    "Project Details": st.column_config.Column(
                        "Project Details",
                        help="Structured project experience (Title, Description, Technologies)",
                        width="large"
                    ),
                    "Notes": st.column_config.TextColumn( # Display notes in shortlisted table
                        "Notes",
                        help="Personal notes for this candidate",
                        width="medium"
                    )
                }
            )
            csv_shortlist_data = manually_shortlisted_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Manually Shortlisted Candidates as CSV",
                data=csv_shortlist_data,
                file_name="manually_shortlisted_candidates.csv",
                mime="text/csv",
                help="Download your manually shortlisted candidates as a CSV file."
            )
        else:
            st.info("No candidates have been manually shortlisted yet. Use the 'Shortlist' checkbox in the 'Comprehensive Candidate Results Table' above to add them.")

        # --- Resume Viewer Section (New) ---
        st.markdown("---")
        st.markdown("## 🔍 Resume Viewer & Detailed Assessment")
        st.caption("Select a candidate to view their raw resume text and full AI assessment.")
        
        # Get candidate names from the full results DataFrame (not filtered)
        candidate_names_for_viewer = ["Select a Candidate"] + sorted(st.session_state['full_results_df']['Candidate Name'].tolist())
        selected_candidate_name_for_viewer = st.selectbox(
            "Choose a candidate to view details:",
            options=candidate_names_for_viewer,
            key="resume_viewer_select"
        )

        if selected_candidate_name_for_viewer != "Select a Candidate":
            selected_candidate_row = st.session_state['full_results_df'][st.session_state['full_results_df']['Candidate Name'] == selected_candidate_name_for_viewer].iloc[0]
            
            st.markdown(f"### Raw Resume Text for {selected_candidate_name_for_viewer}")
            st.text_area("Resume Content", selected_candidate_row['Resume Raw Text'], height=400, disabled=True)
            
            st.markdown(f"### Detailed AI Assessment for {selected_candidate_name_for_viewer}")
            st.markdown(selected_candidate_row['Detailed HR Assessment'])

            # Display Categorized Matched Skills for the selected candidate
            st.markdown("#### Matched Skills Breakdown:")
            if selected_candidate_row['Matched Keywords (Categorized)']:
                for category, skills in selected_candidate_row['Matched Keywords (Categorized)'].items():
                    st.write(f"**{category}:** {', '.join(skills)}")
            else:
                st.write("No categorized matched skills found.")

            # Display Categorized Missing Skills for the selected candidate
            st.markdown("#### Missing Skills Breakdown (from JD):")
            # Need to re-calculate missing skills based on JD's categorized skills and candidate's raw skills
            # This logic is already in the main loop, can reuse the stored values
            if selected_candidate_row['Missing Skills'].strip(): # Check if string is not empty
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

            # Action button for the selected candidate
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
