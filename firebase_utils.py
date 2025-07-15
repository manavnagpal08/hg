import requests
import json
from datetime import datetime
import streamlit as st # Only for st.error, consider removing for pure utility

# --- Firebase REST API Configuration ---
# IMPORTANT: Leave FIREBASE_WEB_API_KEY as an empty string. Canvas will provide it at runtime.
FIREBASE_WEB_API_KEY = ""
# Use __app_id from the Canvas environment for the project ID if available, otherwise use a default
FIREBASE_PROJECT_ID = globals().get('__app_id', 'screenerproapp')
FIRESTORE_BASE_URL = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents"
FIREBASE_AUTH_SIGNUP_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_WEB_API_KEY}"
FIREBASE_AUTH_SIGNIN_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
FIREBASE_AUTH_RESET_PASSWORD_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={FIREBASE_WEB_API_KEY}"
FIREBASE_AUTH_LOOKUP_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={FIREBASE_WEB_API_KEY}"
FIREBASE_AUTH_UPDATE_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:update?key={FIREBASE_WEB_API_KEY}"

# --- Helper functions for Firebase REST API data conversion ---
def to_firestore_format(data: dict) -> dict:
    """Converts a Python dictionary to Firestore REST API 'fields' format."""
    fields = {}
    for key, value in data.items():
        if isinstance(value, str):
            fields[key] = {"stringValue": value}
        elif isinstance(value, int):
            fields[key] = {"integerValue": str(value)} # Firestore expects string for integerValue
        elif isinstance(value, float):
            fields[key] = {"doubleValue": value}
        elif isinstance(value, bool):
            fields[key] = {"booleanValue": value}
        elif isinstance(value, datetime):
            fields[key] = {"timestampValue": value.isoformat(timespec='milliseconds') + "Z"} # ISO 8601 with 'Z' for UTC, milliseconds precision
        elif isinstance(value, list):
            array_values = []
            for item in value:
                if isinstance(item, str):
                    array_values.append({"stringValue": item})
                elif isinstance(item, int):
                    array_values.append({"integerValue": str(item)})
                elif isinstance(item, float):
                    array_values.append({"doubleValue": item})
                elif isinstance(item, bool):
                    array_values.append({"booleanValue": item})
                elif isinstance(item, dict):
                    # Recursively convert nested dictionaries
                    array_values.append({"mapValue": {"fields": to_firestore_format(item)['fields']}})
                else:
                    array_values.append({"stringValue": str(item)}) # Fallback for other types
            fields[key] = {"arrayValue": {"values": array_values}}
        elif isinstance(value, dict):
            # Recursively convert nested dictionaries
            fields[key] = {"mapValue": {"fields": to_firestore_format(value)['fields']}}
        elif value is None:
            fields[key] = {"nullValue": None}
        else:
            fields[key] = {"stringValue": str(value)} # Catch-all for other types
    return {"fields": fields}

def from_firestore_format(firestore_data: dict) -> dict:
    """Converts Firestore REST API 'fields' format to a Python dictionary."""
    data = {}
    if "fields" not in firestore_data:
        return data # Or raise an error if expected
    
    for key, value_obj in firestore_data["fields"].items():
        if "stringValue" in value_obj:
            data[key] = value_obj["stringValue"]
        elif "integerValue" in value_obj:
            data[key] = int(value_obj["integerValue"])
        elif "doubleValue" in value_obj:
            data[key] = float(value_obj["doubleValue"])
        elif "booleanValue" in value_obj:
            data[key] = value_obj["booleanValue"]
        elif "timestampValue" in value_obj:
            try:
                # Remove 'Z' and parse, then localize if needed (though storing as UTC is best)
                data[key] = datetime.fromisoformat(value_obj["timestampValue"].replace('Z', ''))
            except ValueError:
                data[key] = value_obj["timestampValue"] # Keep as string if parsing fails
        elif "arrayValue" in value_obj and "values" in value_obj["arrayValue"]:
            # Handle array values, including nested maps
            data[key] = [
                from_firestore_format({"fields": item["mapValue"]["fields"]}) if "mapValue" in item else
                (item["stringValue"] if "stringValue" in item else
                 (int(item["integerValue"]) if "integerValue" in item else
                  (float(item["doubleValue"]) if "doubleValue" in item else
                   (item["booleanValue"] if "booleanValue" in item else
                    (datetime.fromisoformat(item["timestampValue"].replace('Z', '')) if "timestampValue" in item else
                     None)))) # Fallback for other types in array
            ) for item in value_obj["arrayValue"]["values"]]
        elif "mapValue" in value_obj and "fields" in value_obj["mapValue"]:
            data[key] = from_firestore_format({"fields": value_obj["mapValue"]["fields"]})
        elif "nullValue" in value_obj:
            data[key] = None
    return data

# --- Firestore Document Management (REST API) ---
def get_firestore_document(collection_path: str, doc_id: str):
    """Fetches a single document from Firestore."""
    url = f"{FIRESTORE_BASE_URL}/{collection_path}/{doc_id}?key={FIREBASE_WEB_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return from_firestore_format(response.json())
    except requests.exceptions.RequestException as e:
        if e.response and e.response.status_code == 404:
            return None # Document not found
        st.error(f"Error fetching document {doc_id}: {e.response.text if e.response else e}") # Using st.error for immediate feedback
        return None

def set_firestore_document(collection_path: str, doc_id: str, data: dict):
    """Sets a document in Firestore using PUT (creates or overwrites)."""
    url = f"{FIRESTORE_BASE_URL}/{collection_path}/{doc_id}?key={FIREBASE_WEB_API_KEY}"
    firestore_data = to_firestore_format(data)
    try:
        res = requests.put(url, json=firestore_data)
        res.raise_for_status()
        return True, res.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Firestore set error for {doc_id}: {e.response.text if e.response else e}") # Using st.error for immediate feedback
        return False, str(e)

def delete_firestore_document(collection_path: str, doc_id: str):
    """Deletes a document from Firestore."""
    url = f"{FIRESTORE_BASE_URL}/{collection_path}/{doc_id}?key={FIREBASE_WEB_API_KEY}"
    try:
        res = requests.delete(url)
        res.raise_for_status()
        return True, res.text
    except requests.exceptions.RequestException as e:
        st.error(f"Firestore delete error for {doc_id}: {e.response.text if e.response else e}") # Using st.error for immediate feedback
        return False, str(e)

def fetch_firestore_collection(collection_path: str, order_by_field: str = None, limit: int = 100):
    """Fetches documents from a Firestore collection."""
    url = f"{FIRESTORE_BASE_URL}/{collection_path}?key={FIREBASE_WEB_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        documents = []
        if 'documents' in data:
            for doc_entry in data['documents']:
                doc_id = doc_entry['name'].split('/')[-1]
                doc_data = from_firestore_format(doc_entry)
                doc_data['id'] = doc_id
                documents.append(doc_data)
        
        if order_by_field and all(order_by_field in doc for doc in documents):
            documents.sort(key=lambda x: x.get(order_by_field), reverse=True)
        
        return documents[:limit]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching collection {collection_path}: {e.response.text if e.response else e}") # Using st.error for immediate feedback
        return []

def log_activity_to_firestore(message: str, user: str):
    """
    Logs an activity with a timestamp to Firestore (via REST API).
    This log is public/common across all companies.
    """
    try:
        collection_url = f"artifacts/{FIREBASE_PROJECT_ID}/public/data/activity_feed"
        # For logging, we'll use a POST request to let Firestore generate the document ID
        # This requires changing the URL slightly for POST requests to a collection
        url = f"{FIRESTORE_BASE_URL}/{collection_url}?key={FIREBASE_WEB_API_KEY}"
        payload = to_firestore_format({
            "message": message,
            "user": user,
            "timestamp": datetime.now()
        })
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error logging activity to Firestore via REST API: {e.response.text if e.response else e}") # Print to console, avoid st.error in utility

