import streamlit as st

def about_us_page():
    """
    Renders the About Us page content.
    """

    st.markdown('<div class="dashboard-header">🏢 About Us</div>', unsafe_allow_html=True)
    st.write("""
    Welcome to **ScreenerPro**, your intelligent partner in modern recruitment.
    We leverage cutting-edge AI to streamline your hiring process,
    helping you identify the best talent quickly and efficiently.

    Whether you're a startup looking to scale or an enterprise managing thousands of applications,
    ScreenerPro simplifies the way you screen resumes with speed, accuracy, and fairness at its core.
    """)

    st.subheader("Our Vision")
    st.write("""
    At ScreenerPro, our vision is to revolutionize talent acquisition by making it
    more data-driven, fair, and efficient. We believe in empowering HR professionals
    and hiring managers with tools that not only save time but also enhance the
    quality of hires, ultimately contributing to organizational success.

    We envision a future where hiring is not just faster but smarter — where
    human potential is matched with opportunity in the most intelligent way possible.
    """)

    st.subheader("Our Mission")
    st.write("""
    Our mission is to provide an intuitive, powerful, and reliable resume screening solution
    that reduces unconscious bias, highlights truly relevant skills, and accelerates the
    shortlisting process. We are committed to continuous innovation, ensuring our platform
    remains at the forefront of AI-powered HR technology.

    We aim to bridge the gap between talent and opportunity by bringing automation,
    transparency, and data science into the heart of recruitment.
    """)

    st.subheader("Our Team")
    st.write("""
    We are a passionate team of AI engineers, data scientists, and HR experts dedicated
    to solving real-world recruitment challenges. Our diverse backgrounds enable us
    to build a product that is both technically robust and deeply understands the
    nuances of human resources.

    Our team brings together experience from leading tech companies and research institutions.
    From designing fair screening algorithms to crafting a delightful user experience,
    every feature we build is driven by real hiring pain points and feedback from users like you.
    """)

    st.subheader("Why Choose ScreenerPro?")
    st.write("""
    ✅ **AI-Powered Screening** – Our machine learning models analyze resumes beyond keywords, focusing on true potential.  
    ✅ **Bias Reduction** – We help companies move toward inclusive hiring practices by minimizing unconscious biases in screening.  
    ✅ **Speed & Accuracy** – Reduce hours of manual shortlisting into minutes without compromising quality.  
    ✅ **Built for HR Teams** – Every tool and dashboard is built with recruiters in mind, ensuring usability and practical value.  
    ✅ **Secure & Compliant** – Your data is always encrypted and processed responsibly in line with global data protection laws.

    At ScreenerPro, you’re not just using a tool — you’re joining a movement to modernize and humanize hiring with technology.
    """)

    st.markdown("---")
    st.write("For inquiries, please use the **'Feedback & Help'** section in the sidebar or contact our support team anytime.")
