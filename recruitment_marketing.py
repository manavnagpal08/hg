import streamlit as st
import requests
from datetime import datetime
import json
import pandas as pd

# --- Helper functions (copied for self-containment) ---
def to_firestore_format(data: dict) -> dict:
    """Converts a Python dictionary to Firestore REST API 'fields' format."""
    fields = {}
    for key, value in data.items():
        if isinstance(value, str):
            fields[key] = {"stringValue": value}
        elif isinstance(value, int):
            fields[key] = {"integerValue": str(value)}
        elif isinstance(value, float):
            fields[key] = {"doubleValue": value}
        elif isinstance(value, bool):
            fields[key] = {"booleanValue": value}
        elif isinstance(value, datetime):
            fields[key] = {"timestampValue": value.isoformat() + "Z"}
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
                    array_values.append({"mapValue": {"fields": to_firestore_format(item)['fields']}})
                else:
                    array_values.append({"stringValue": str(item)})
            fields[key] = {"arrayValue": {"values": array_values}}
        elif isinstance(value, dict):
            fields[key] = {"mapValue": {"fields": to_firestore_format(value)['fields']}}
        elif value is None:
            fields[key] = {"nullValue": None}
        else:
            fields[key] = {"stringValue": str(value)}
    return {"fields": fields}

def from_firestore_format(firestore_data: dict) -> dict:
    """Converts Firestore REST API 'fields' format to a Python dictionary."""
    data = {}
    if "fields" not in firestore_data:
        return data
    
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
                data[key] = datetime.fromisoformat(value_obj["timestampValue"].replace('Z', ''))
            except ValueError:
                data[key] = value_obj["timestampValue"]
        elif "arrayValue" in value_obj and "values" in value_obj["arrayValue"]:
            data[key] = [from_firestore_format({"fields": {"_": item}})["_"] if "mapValue" not in item else from_firestore_format({"fields": item["mapValue"]["fields"]}) for item in value_obj["arrayValue"]["values"]]
        elif "mapValue" in value_obj and "fields" in value_obj["mapValue"]:
            data[key] = from_firestore_format({"fields": value_obj["mapValue"]["fields"]})
        elif "nullValue" in value_obj:
            data[key] = None
    return data

def log_activity(message: str, user: str, FIREBASE_WEB_API_KEY: str, FIRESTORE_BASE_URL: str, app_id: str):
    """
    Logs an activity with a timestamp to Firestore (via REST API) and session state.
    This log is public/common across all companies.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {user}: {message}"

    if 'activity_log' not in st.session_state:
        st.session_state.activity_log = []
    st.session_state.activity_log.insert(0, log_entry)
    st.session_state.activity_log = st.session_state.activity_log[:50]

    try:
        collection_url = f"{FIRESTORE_BASE_URL}/artifacts/{app_id}/public/data/activity_feed"
        payload = to_firestore_format({
            "message": message,
            "user": user,
            "timestamp": datetime.now()
        })
        response = requests.post(collection_url, json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error logging activity to Firestore via REST API: {e}")

def fetch_collection_data(collection_path: str, FIREBASE_WEB_API_KEY: str, FIRESTORE_BASE_URL: str, order_by_field: str = None, limit: int = 20):
    """Fetches documents from a Firestore collection via REST API."""
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
        st.error(f"Error fetching data from {collection_path} via REST API: {e}")
        return []

def add_document_to_firestore_collection(collection_path, data, api_key, base_url):
    """Adds a new document to a Firestore collection (Firestore assigns ID)."""
    url = f"{base_url}/{collection_path}?key={api_key}"
    firestore_data = to_firestore_format(data)
    try:
        res = requests.post(url, json=firestore_data)
        res.raise_for_status()
        return True, res.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Firestore add error: {e}")
        return False, str(e)

def update_document_in_firestore(collection_path, doc_id, data, api_key, base_url):
    """Updates a document in Firestore using PATCH."""
    url = f"{base_url}/{collection_path}/{doc_id}?key={api_key}"
    update_mask_fields = ",".join(data.keys())
    url_with_mask = f"{url}&updateMask.fieldPaths={update_mask_fields}"
    
    firestore_data = to_firestore_format(data)
    try:
        res = requests.patch(url_with_mask, json=firestore_data)
        res.raise_for_status()
        return True, res.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Firestore update error: {e}")
        return False, str(e)

def delete_document_from_firestore(collection_path, doc_id, api_key, base_url):
    """Deletes a document from Firestore."""
    url = f"{base_url}/{collection_path}/{doc_id}?key={api_key}"
    try:
        res = requests.delete(url)
        res.raise_for_status()
        return True, res.text
    except requests.exceptions.RequestException as e:
        st.error(f"Firestore delete error: {e}")
        return False, str(e)

# --- Main Recruitment Marketing Page Function ---
def recruitment_marketing_page(app_id: str, FIREBASE_WEB_API_KEY: str, FIRESTORE_BASE_URL: str):
    st.markdown('<div class="dashboard-header">📢 Recruitment Marketing & Employer Branding</div>', unsafe_allow_html=True)
    st.write("Manage your company's public image as an employer and attract top talent.")

    current_username = st.session_state.get('username', 'Anonymous User')
    user_company = st.session_state.get('user_company', 'default_company').replace(' ', '_').lower()

    st.info(f"You are managing branding for: **{st.session_state.get('user_company', 'N/A')}**")
    st.write(f"**DEBUG: Current Company for Data Isolation:** `{user_company}`")
    st.markdown("---")

    # Tabs for different features
    tab_career_page, tab_social_media, tab_testimonials, tab_analytics = st.tabs([
        "🌐 Career Page Content", "📱 Social Media Posts", "🌟 Candidate Testimonials", "📈 Campaign Analytics (Mock)"
    ])

    with tab_career_page:
        st.subheader("Career Page Content Management")
        st.info(f"Content here is for your company's career page, visible to potential candidates. Data is isolated for **{st.session_state.get('user_company', 'your company')}**.")

        # Initialize session state for career page content
        if 'career_page_content' not in st.session_state:
            st.session_state.career_page_content = []
        if 'career_page_needs_refresh' not in st.session_state:
            st.session_state.career_page_needs_refresh = True

        with st.form("career_page_form", clear_on_submit=True):
            content_title = st.text_input("Content Title (e.g., 'Our Culture', 'Why Work Here'):", key="cp_title")
            content_type = st.selectbox("Content Type:", ["Text", "Image URL", "Video URL"], key="cp_type")
            content_value = st.text_area("Content (Text or URL):", key="cp_value")
            add_content_button = st.form_submit_button("Add/Update Content")

            if add_content_button:
                if content_title and content_value:
                    try:
                        collection_path = f"artifacts/{app_id}/companies/{user_company}/recruitment_marketing/career_page_content"
                        # Use title as doc_id for easy updates
                        doc_id = content_title.replace(' ', '_').lower()
                        content_data = {
                            "title": content_title,
                            "type": content_type,
                            "value": content_value,
                            "created_by": current_username,
                            "created_at": datetime.now()
                        }
                        success, response_data = update_document_in_firestore(
                            collection_path, doc_id, content_data, FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL
                        )
                        if success:
                            st.success(f"Career page content '{content_title}' added/updated successfully!")
                            log_activity(f"updated career page content '{content_title}'.", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                            st.session_state.career_page_needs_refresh = True
                            st.rerun()
                        else:
                            st.error(f"Error adding/updating content: {response_data}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
                else:
                    st.warning("Please provide a title and content value.")
        
        st.markdown("---")
        st.subheader("Current Career Page Content")

        if st.button("Refresh Career Page Content", key="refresh_cp_button") or st.session_state.career_page_needs_refresh:
            st.session_state.career_page_content = fetch_collection_data(
                f"artifacts/{app_id}/companies/{user_company}/recruitment_marketing/career_page_content",
                FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY,
                FIRESTORE_BASE_URL=FIRESTORE_BASE_URL,
                order_by_field="created_at",
                limit=10
            )
            st.session_state.career_page_needs_refresh = False

        if st.session_state.career_page_content:
            for content_item in st.session_state.career_page_content:
                timestamp_obj = content_item.get('created_at')
                timestamp_str = timestamp_obj.strftime("%Y-%m-%d %H:%M") if isinstance(timestamp_obj, datetime) else str(timestamp_obj)

                st.markdown(f"""
                <div style="background-color: {'#3A3A3A' if st.session_state.get('dark_mode_main') else '#f0f2f6'}; padding: 15px; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <h5 style="color: {'#00cec9' if st.session_state.get('dark_mode_main') else '#00cec9'}; margin-bottom: 5px;">
                        {content_item.get('title', 'No Title')} ({content_item.get('type', 'N/A')})
                    </h5>
                    <p style="font-size: 0.9em; color: {'#BBBBBB' if st.session_state.get('dark_mode_main') else '#666'}; margin-bottom: 5px;">
                        Added by: **{content_item.get('created_by', 'Unknown')}** at {timestamp_str}
                    </p>
                """, unsafe_allow_html=True)
                
                if content_item.get('type') == "Text":
                    st.write(content_item.get('value', ''))
                elif content_item.get('type') == "Image URL":
                    st.image(content_item.get('value', 'https://placehold.co/600x200?text=Image+Placeholder'), caption=content_item.get('title', ''))
                elif content_item.get('type') == "Video URL":
                    st.video(content_item.get('value', ''))
                
                if st.button(f"Delete '{content_item['title']}'", key=f"delete_cp_{content_item['id']}"):
                    try:
                        collection_path = f"artifacts/{app_id}/companies/{user_company}/recruitment_marketing/career_page_content"
                        success, response_data = delete_document_from_firestore(
                            collection_path, content_item['id'], FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL
                        )
                        if success:
                            st.success(f"Content '{content_item['title']}' deleted.")
                            log_activity(f"deleted career page content '{content_item['title']}'.", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                            st.session_state.career_page_needs_refresh = True
                            st.rerun()
                        else:
                            st.error(f"Error deleting content: {response_data}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No career page content added yet. Add some above!")


    with tab_social_media:
        st.subheader("Social Media Job Postings (Mock)")
        st.info(f"Manage mock social media posts for job openings. This is a placeholder; actual integration with social media platforms is not supported. Data is isolated for **{st.session_state.get('user_company', 'your company')}**.")

        # Initialize session state for social media posts
        if 'social_media_posts' not in st.session_state:
            st.session_state.social_media_posts = []
        if 'social_media_needs_refresh' not in st.session_state:
            st.session_state.social_media_needs_refresh = True

        with st.form("social_media_form", clear_on_submit=True):
            post_platform = st.selectbox("Platform:", ["LinkedIn", "Twitter (X)", "Facebook", "Instagram"], key="sm_platform")
            post_content = st.text_area("Post Content:", key="sm_content")
            post_job_link = st.text_input("Link to Job Posting (Optional):", key="sm_job_link")
            post_date = st.date_input("Scheduled Post Date:", key="sm_post_date", value=datetime.now().date())
            add_post_button = st.form_submit_button("Schedule Post")

            if add_post_button:
                if post_content:
                    try:
                        collection_path = f"artifacts/{app_id}/companies/{user_company}/recruitment_marketing/social_media_posts"
                        post_data = {
                            "platform": post_platform,
                            "content": post_content,
                            "job_link": post_job_link,
                            "scheduled_date": str(post_date),
                            "posted_by": current_username,
                            "created_at": datetime.now()
                        }
                        success, response_data = add_document_to_firestore_collection(
                            collection_path, post_data, FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL
                        )
                        if success:
                            st.success(f"Mock social media post scheduled for {post_platform}!")
                            log_activity(f"scheduled a mock social media post for {post_platform}.", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                            st.session_state.social_media_needs_refresh = True
                            st.rerun()
                        else:
                            st.error(f"Error scheduling post: {response_data}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
                else:
                    st.warning("Please provide content for the post.")
        
        st.markdown("---")
        st.subheader("Scheduled Social Media Posts")

        if st.button("Refresh Social Media Posts", key="refresh_sm_button") or st.session_state.social_media_needs_refresh:
            st.session_state.social_media_posts = fetch_collection_data(
                f"artifacts/{app_id}/companies/{user_company}/recruitment_marketing/social_media_posts",
                FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY,
                FIRESTORE_BASE_URL=FIRESTORE_BASE_URL,
                order_by_field="scheduled_date",
                limit=20
            )
            st.session_state.social_media_needs_refresh = False

        if st.session_state.social_media_posts:
            for post in st.session_state.social_media_posts:
                timestamp_obj = post.get('created_at')
                timestamp_str = timestamp_obj.strftime("%Y-%m-%d %H:%M") if isinstance(timestamp_obj, datetime) else str(timestamp_obj)
                scheduled_date_str = post.get('scheduled_date', 'N/A')

                st.markdown(f"""
                <div style="background-color: {'#3A3A3A' if st.session_state.get('dark_mode_main') else '#f0f2f6'}; padding: 15px; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <h5 style="color: {'#00cec9' if st.session_state.get('dark_mode_main') else '#00cec9'}; margin-bottom: 5px;">
                        {post.get('platform', 'N/A')} Post (Scheduled: {scheduled_date_str})
                    </h5>
                    <p style="font-size: 1.0em; color: {'#E0E0E0' if st.session_state.get('dark_mode_main') else '#333'};">
                        {post.get('content', 'No content')}
                    </p>
                    {"<p><a href='" + post['job_link'] + "' target='_blank' style='color: #00cec9;'>View Job Link</a></p>" if post.get('job_link') else ""}
                    <p style="font-size: 0.8em; color: {'#999999' if st.session_state.get('dark_mode_main') else '#888'}; margin-top: 5px;">
                        Scheduled by: {post.get('posted_by', 'Unknown')} at {timestamp_str}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No social media posts scheduled yet. Schedule one above!")

    with tab_testimonials:
        st.subheader("Candidate Testimonials")
        st.info(f"Showcase positive feedback from candidates to enhance your employer brand. Data is isolated for **{st.session_state.get('user_company', 'your company')}**.")

        # Initialize session state for testimonials
        if 'candidate_testimonials' not in st.session_state:
            st.session_state.candidate_testimonials = []
        if 'testimonials_needs_refresh' not in st.session_state:
            st.session_state.testimonials_needs_refresh = True

        with st.form("testimonial_form", clear_on_submit=True):
            testimonial_author = st.text_input("Candidate Name:", key="test_author")
            testimonial_role = st.text_input("Role/Position (e.g., 'Software Engineer Candidate'):", key="test_role")
            testimonial_text = st.text_area("Testimonial Text:", key="test_text")
            add_testimonial_button = st.form_submit_button("Add Testimonial")

            if add_testimonial_button:
                if testimonial_author and testimonial_text:
                    try:
                        collection_path = f"artifacts/{app_id}/companies/{user_company}/recruitment_marketing/candidate_testimonials"
                        testimonial_data = {
                            "author": testimonial_author,
                            "role": testimonial_role,
                            "text": testimonial_text,
                            "added_by": current_username,
                            "created_at": datetime.now()
                        }
                        success, response_data = add_document_to_firestore_collection(
                            collection_path, testimonial_data, FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL
                        )
                        if success:
                            st.success("Testimonial added successfully!")
                            log_activity(f"added a candidate testimonial from '{testimonial_author}'.", user=current_username, FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY, FIRESTORE_BASE_URL=FIRESTORE_BASE_URL, app_id=app_id)
                            st.session_state.testimonials_needs_refresh = True
                            st.rerun()
                        else:
                            st.error(f"Error adding testimonial: {response_data}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
                else:
                    st.warning("Please provide candidate name and testimonial text.")
        
        st.markdown("---")
        st.subheader("Published Testimonials")

        if st.button("Refresh Testimonials", key="refresh_test_button") or st.session_state.testimonials_needs_refresh:
            st.session_state.candidate_testimonials = fetch_collection_data(
                f"artifacts/{app_id}/companies/{user_company}/recruitment_marketing/candidate_testimonials",
                FIREBASE_WEB_API_KEY=FIREBASE_WEB_API_KEY,
                FIRESTORE_BASE_URL=FIRESTORE_BASE_URL,
                order_by_field="created_at",
                limit=10
            )
            st.session_state.testimonials_needs_refresh = False

        if st.session_state.candidate_testimonials:
            for testimonial in st.session_state.candidate_testimonials:
                timestamp_obj = testimonial.get('created_at')
                timestamp_str = timestamp_obj.strftime("%Y-%m-%d %H:%M") if isinstance(timestamp_obj, datetime) else str(timestamp_obj)

                st.markdown(f"""
                <div style="background-color: {'#3A3A3A' if st.session_state.get('dark_mode_main') else '#f0f2f6'}; padding: 15px; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <p style="font-size: 1.1em; color: {'#E0E0E0' if st.session_state.get('dark_mode_main') else '#333'}; font-style: italic;">
                        "{testimonial.get('text', 'No testimonial text.')}"
                    </p>
                    <p style="font-size: 0.9em; color: {'#BBBBBB' if st.session_state.get('dark_mode_main') else '#666'}; text-align: right;">
                        - **{testimonial.get('author', 'Unknown')}** ({testimonial.get('role', 'Candidate')})
                    </p>
                    <p style="font-size: 0.8em; color: {'#999999' if st.session_state.get('dark_mode_main') else '#888'}; margin-top: 5px;">
                        Added by: {testimonial.get('added_by', 'Unknown')} at {timestamp_str}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No candidate testimonials added yet. Add some above!")

    with tab_analytics:
        st.subheader("Recruitment Campaign Analytics (Mock)")
        st.info("This section provides mock analytics for your recruitment marketing efforts. A full implementation would integrate with advertising platforms and career sites to pull real-time data.")
        st.write(f"Mock data for **{st.session_state.get('user_company', 'your company')}**'s recent campaigns:")

        mock_campaign_data = {
            "Campaign Name": ["Spring Hiring Drive", "Tech Talent Push", "Diversity Initiative"],
            "Impressions": [150000, 200000, 80000],
            "Clicks": [3500, 6000, 1200],
            "Applications": [150, 250, 60],
            "Hires": [10, 15, 5],
            "Cost": [5000, 7500, 3000],
        }
        campaign_df = pd.DataFrame(mock_campaign_data)
        
        campaign_df['CTR (%)'] = (campaign_df['Clicks'] / campaign_df['Impressions'] * 100).round(2)
        campaign_df['Conversion Rate (%)'] = (campaign_df['Applications'] / campaign_df['Clicks'] * 100).round(2)
        campaign_df['Cost Per Hire'] = (campaign_df['Cost'] / campaign_df['Hires']).round(2)

        st.dataframe(campaign_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Key Metrics Over Time (Mock Chart)")
        st.write("Visualize trends in your recruitment efforts.")

        mock_trend_data = {
            "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
            "Applications": [100, 120, 150, 130, 180, 200],
            "Hires": [5, 7, 8, 6, 10, 12],
            "Website Visits": [5000, 5500, 6200, 5800, 7000, 7500]
        }
        trend_df = pd.DataFrame(mock_trend_data)
        
        st.line_chart(trend_df.set_index('Month'))

