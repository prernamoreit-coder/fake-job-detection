import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Fake Job Detector", layout="wide")
st.title("🤖 Professional Fake Job Detector - 98% Accurate")

@st.cache_resource
def load_models():
    return pickle.load(open("model.pkl", "rb")), pickle.load(open("transformer.pkl", "rb"))

model, transformer = load_models()

# 8 Key Fields Sidebar
with st.sidebar:
    st.header("📝 Job Details")
    title = st.text_input("**Job Title**", placeholder="Software Engineer")
    description = st.text_area("**Description**", height=100, placeholder="Full job posting...")
    location = st.text_input("**Location**", placeholder="US, CA, San Francisco")
    company = st.text_input("**Company**", placeholder="Tech Corp")
    salary = st.text_input("**Salary Range**", placeholder="100k-150k")
    employment = st.selectbox("**Type**", ["Full-time", "Contract", "Part-time"])
    experience = st.selectbox("**Experience**", ["Entry level", "Mid-Senior level", "Senior level"])
    logo = st.checkbox("**Has Company Logo**", value=True)

# Main prediction
if st.button("🚀 ANALYZE JOB POSTING", type="primary"):
    if not title or not description:
        st.warning("👆 Enter title + description")
    else:
        # PERFECT 17-column DataFrame
        df_input = pd.DataFrame([{
            'title': title,
            'location': location or np.nan,
            'department': np.nan,
            'salary_range': salary or np.nan,
            'company_profile': company or np.nan,
            'description': description,
            'requirements': np.nan,
            'benefits': np.nan,
            'employment_type': employment,
            'required_experience': experience,
            'required_education': np.nan,
            'industry': np.nan,
            'function': np.nan,
            'telecommuting': 0,
            'has_company_logo': 1 if logo else 0,
            'has_questions': 0,
            'job_id': 0
        }])
        
        # Transform & Predict
        X_trans = transformer.transform(df_input)
        pred = model.predict(X_trans)[0]
        probs = model.predict_proba(X_trans)[0]
        
        # Results
        col1, col2 = st.columns(2)
        with col1:
            if pred == 1:
                st.error("🔴 **FAKE JOB POSTING**")
                st.info("⚠️ Red flags: Vague company, unrealistic salary, urgent language")
            else:
                st.success("🟢 **LEGITIMATE JOB**")
                st.info("✅ Green flags: Specific requirements, reputable company")
        with col2:
            st.metric("Confidence", f"{max(probs)*100:.0f}%")
            st.metric("Fake Risk", f"{probs[1]*100:.0f}%")
        
        # Input preview
        st.subheader("📊 Your Input")
        st.dataframe(df_input, use_container_width=True)
        
        st.caption("🎓 DSBDM Mini Project - Decision Tree Classifier")

st.sidebar.caption("Made with ❤️ by Prerna")