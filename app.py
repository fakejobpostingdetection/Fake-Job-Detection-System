import streamlit as st
import joblib
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("Groq API key not found.")
    st.stop()

client = Groq(api_key=api_key)


# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)
def load_css(css_file):
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles/style.css")
import base64

def add_bg(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(f"""
    <style>

    .stApp {{
        background:
        linear-gradient(
        rgba(5,15,35,.40),
        rgba(5,15,35,.55)
        ),
        url("data:image/jpg;base64,{encoded}");

        background-size:110%;;
        background-position:center center;
        background-repeat:no-repeat;
        background-attachment:fixed;
    }}

    </style>
    """, unsafe_allow_html=True)

add_bg("images/bg.jpeg")
# -----------------------------------
# Load Model and Vectorizer
# -----------------------------------

model = joblib.load("models/best_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# -----------------------------------
# Sidebar
# -----------------------------------

st.sidebar.title("Project Information")

st.sidebar.success("Model Used : XGBoost")

st.sidebar.info("Accuracy : 98.6%")

st.sidebar.write("")

st.sidebar.subheader("Developer")

st.sidebar.write("Nancy,Pari Singh")

st.sidebar.write("")

st.sidebar.subheader("Technology")

st.sidebar.write("Machine Learning")

st.sidebar.write("TF-IDF")

st.sidebar.write("XGBoost")

st.sidebar.write("Streamlit")

# -----------------------------------
# Main Title
# -----------------------------------

st.title("Fake Job Detection System")

st.markdown("""
Detect whether a job posting is **Fake** or **Genuine** using **Machine Learning**.

This project uses **TF-IDF + XGBoost** to classify job postings.
""")

st.write("---")

# -----------------------------------
# Input Fields
# -----------------------------------

title = st.text_input("Job Title")

company = st.text_input("Company Profile")

description = st.text_area(
    "Job Description",
    height=170
)

requirements = st.text_area(
    "Requirements",
    height=120
)

benefits = st.text_area(
    "Benefits",
    height=120
)

st.write("---")

# -----------------------------------
# Prediction
# -----------------------------------

if st.button("Predict Job"):

    text = (
        title + " " +
        company + " " +
        description + " " +
        requirements + " " +
        benefits
    )

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]

    st.write("")

    if prediction == 1:

        st.error(" FAKE JOB DETECTED")

        st.metric(
            label="Risk Level",
            value="HIGH"
        )
        st.write("---")
        st.subheader(" AI Fraud Analysis")

        prompt = f"""
        You are a cyber fraud expert.

        Analyze the following job posting.

        Job Title:
        {title}

        Company:
        {company}

        Description:
        {description}

        Requirements:
        {requirements}

        Benefits:
        {benefits}

        The ML model has predicted that this job is FAKE.

        Generate a professional report with the following headings:

        1. Fraud Summary

        2. Red Flags (bullet points)

        3. Possible Scam Type

        4. Risk Level

        5. Safety Recommendations

        Keep the response simple and professional.
        """

        try:
            response = client.chat.completions.create(

                model="llama-3.1-8b-instant",

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0.3

            )

            st.markdown(
                response.choices[0].message.content
            )

        except Exception as e:
            st.error(f"Groq Error: {e}")

        st.warning("""
### Warning

This job posting appears to be suspicious.

Possible reasons:

• Unrealistic salary

• Missing company information

• Instant joining

• No experience required

• Contact through WhatsApp

Always verify the company before applying.
""")

    else:

        st.success("GENUINE JOB")
        st.write("---")
        st.subheader("AI Analysis")

        prompt = f"""
        You are a recruitment expert.

        Analyze the following job posting.

        Job Title:
        {title}

        Company:
        {company}

        Description:
        {description}

        Requirements:
        {requirements}

        Benefits:
        {benefits}

        The ML model predicts that this job is GENUINE.

        Generate a report with:

        1. Job Summary

        2. Positive Indicators

        3. Risk Level

        4. Advice for Applicant

        Keep it professional and concise.
        """

        try:
            response = client.chat.completions.create(

                model="llama-3.1-8b-instant",

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0.3

            )

            st.markdown(
                response.choices[0].message.content
            )

        except Exception as e:
            st.error(f"Groq Error: {e}")

        st.metric(
                label="Risk Level",
                 value="LOW"
            )

        st.info("""
        This job appears to be legitimate based on the trained Machine Learning model.

        Still verify the company before sharing personal documents.
        """)

        st.write("---")

        st.caption(
            "Fake Job Detection System | AI & Machine Learning Project | 2026"
        )
# -----------------------------------
# Dashboard
# -----------------------------------

st.write("---")

st.header("Dataset Statistics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Jobs", "17,880")
col2.metric("Real Jobs", "17,014")
col3.metric("Fake Jobs", "866")

st.write("---")

st.header("Dataset Visualizations")

tab1, tab2, tab3 = st.tabs([
    "Distribution",
    "WordCloud",
    "Confusion Matrix"
])

with tab1:

    st.subheader("Fraud Distribution")

    st.image(
        "images/fraud_distribution_bar.png",
        use_container_width=True
    )

    st.image(
        "images/fraud_distribution_pie.png",
        use_container_width=True
    )

with tab2:

    st.subheader("Fake Job WordCloud")

    st.image(
        "images/fake_wordcloud.png",
        use_container_width=True
    )

    st.subheader("Real Job WordCloud")

    st.image(
        "images/real_wordcloud.png",
        use_container_width=True
    )

with tab3:

    st.subheader("Confusion Matrix")

    st.image(
        "images/confusion_matrix.png",
        use_container_width=True
    )

st.write("---")

st.header("Project Summary")

st.success("""
✔ Dataset : Fake Job Postings

✔ Algorithm : XGBoost

✔ Feature Extraction : TF-IDF

✔ Accuracy : 98.6%

✔ Developed using Python & Streamlit
""")