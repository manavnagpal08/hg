import streamlit as st

def privacy_policy_page():
    """
    Renders the Privacy Policy & Terms page content.
    """

    st.markdown('<div class="dashboard-header">⚖️ Privacy Policy & Terms</div>', unsafe_allow_html=True)

    st.subheader("Privacy Policy")
    st.write("""
    **Effective Date: July 17, 2025**

    At **ScreenerPro**, your trust is our top priority. We are committed to handling your data responsibly, transparently, and securely. This Privacy Policy explains how we collect, use, and protect the information you provide while using our resume screening platform.

    ### 🔍 What Information Do We Collect?

    1. **Basic User Information**  
       When you create an account, we collect your **email address** and **company name**. This helps us personalize your dashboard and manage your sessions securely.

    2. **Resume & Job Description Data**  
       When you upload resumes or job descriptions:
       - We process the text content using AI to extract relevant skills, education, experience, etc.
       - We do **not permanently store the full resume text**.
       - Only essential, anonymized metadata (like matching score, top skills) is saved to ensure session continuity.

    3. **Usage & Interaction Data**  
       To improve our service, we collect data like:
       - Pages visited or features used
       - Time spent on each feature
       - System performance metrics
       This allows us to optimize user experience and recommend improvements.

    ### ✅ How Do We Use Your Information?

    Your data helps us:
    - Deliver resume screening and matching insights via our dashboard
    - Provide personalized performance charts, visual feedback, and reporting
    - Improve our AI models with anonymous learning techniques (we **never use candidate names** or contact info for training)
    - Send you helpful updates about new features or critical issues (we avoid spam or sales messages)

    ### 🔐 Data Security

    We follow strict security standards:
    - Passwords are hashed and encrypted.
    - Data is transmitted over **secure HTTPS**.
    - All resume data is processed either in-memory or stored temporarily with access controls in place.
    - Our team is trained in safe data handling practices.

    ### 🗂️ Data Retention & Deletion

    - Your dashboard data (screening scores, uploaded summaries) remains available while your account is active.
    - You may request complete deletion of your account and associated data anytime via the **Feedback & Help** section.
    - We do not retain unused raw resume files or personal details for longer than necessary.

    ### 🧭 Your Rights

    You have the right to:
    - Access the data we store about your account
    - Correct or update inaccurate data
    - Delete all stored data permanently

    We will respond promptly to your requests, usually within 3–5 working days.

    ### 🔁 Updates to This Policy

    We may update this policy as ScreenerPro evolves. Major changes will be announced in the app, and the latest version will always be available on this page.
    """)

    st.subheader("Terms of Service")
    st.write("""
    **Effective Date: July 17, 2025**

    Welcome to **ScreenerPro**! These Terms of Service outline the rules and responsibilities that come with using our platform. By accessing ScreenerPro, you agree to these terms.

    ### 1. 📘 Acceptance of Terms

    By signing up or using ScreenerPro, you confirm that you’ve read and accepted these Terms and our Privacy Policy. If you have questions or concerns, feel free to contact us before continuing — we’re here to support you.

    ### 2. 🚀 Use of the Service

    ScreenerPro is an AI-based resume screening and analysis tool designed for recruiters, HR teams, and hiring managers. You agree to:
    - Use the platform for lawful, professional purposes only.
    - Ensure uploaded data (resumes, job descriptions) is obtained and shared in compliance with data privacy laws such as **GDPR**, **CCPA**, etc.

    ### 3. 👤 User Accounts

    - You must provide accurate information during signup.
    - Keep your login credentials confidential — you're responsible for activities under your account.
    - Report any unauthorized access or suspicious activity immediately via our helpdesk.

    ### 4. 🧠 Intellectual Property

    - ScreenerPro owns all rights to the platform, AI models, design, and tools.
    - You may use the platform's features but may not copy, resell, or reproduce its core systems.
    - Content **you upload** (resumes, JD files) remains your property — we **do not claim ownership** of your data.

    ### 5. ⚠️ Disclaimer on AI Suggestions

    ScreenerPro provides AI-based resume insights, but these are meant to **support** your decisions — not replace human judgment.  
    You should always review candidate information manually before making final hiring decisions.

    ### 6. ⚖️ Governing Law

    These Terms shall be governed in accordance with the laws of **India** (or your local jurisdiction, where applicable).

    ### 7. 🛟 Need Help?

    Have questions about how your data is handled or how our tools work?  
    Use the **"Feedback & Help"** button in the app or email our team directly — we’re happy to assist.

    ---
    We built ScreenerPro to make hiring faster, smarter, and fairer.  
    Thank you for trusting us — we’ll keep working hard to earn that trust every day.
    """)

    st.markdown("---")
    st.write("Your continued use of **ScreenerPro** confirms your agreement with these updated policies. We’re committed to transparency and your data safety.")
