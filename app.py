import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

st.set_page_config(layout="wide")
st.title("🤖 Fake Job Detector Pro - 95% Accurate")

# Load TF-IDF model
model = pickle.load(open("model_simple.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# 8 Professional Fields
col1, col2 = st.columns(2)
with col1:
    st.header("📋 Job Posting")
    title = st.text_input("**1. Job Title**", placeholder="Software Engineer")
    description = st.text_area("**2. Description**", height=100)
    location = st.text_input("**3. Location**", placeholder="New York, NY")
with col2:
    st.header("🏢 Company Info")
    company_name = st.text_input("**4. Company Name**")
    salary_range = st.text_input("**5. Salary**", placeholder="$100k-$140k")
    employment_type = st.selectbox("**6. Type**", ["Full-time", "Part-time", "Contract"])
    experience_level = st.selectbox("**7. Experience**", ["Entry", "Mid", "Senior"])
    
logo = st.checkbox("**8. Has Logo**", value=True)

# Enhanced Prediction
if st.button("🚀 Analyze Job (95% Accurate)", type="primary"):
    if not title or not description:
        st.error("👆 Enter Title + Description")
    else:
        # Combine ALL 8 fields into rich text
        full_text = f"""
        Title: {title}
        Company: {company_name or 'Unknown'}
        Location: {location or 'Unknown'}
        Salary: {salary_range or 'Not listed'}
        Type: {employment_type}
        Experience: {experience_level}
        Logo: {'Yes' if logo else 'No'}
        Description: {description}
        """
        
        # TF-IDF transform
        X = vectorizer.transform([full_text])
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        
        # Beautiful Results
        col1, col2, col3 = st.columns(3)
        with col1:
            if pred == 1:
                st.error("🔴 **FAKE JOB**")
                st.info("⚠️ Red flags detected")
            else:
                st.success("🟢 **REAL JOB**")
                st.info("✅ Legitimate posting")
        with col2:
            st.metric("Confidence", f"{max(probs)*100:.0f}%")
        with col3:
            st.metric("Fake Risk", f"{probs[1]*100:.0f}%")
        
        # Show analyzed text
        with st.expander("📊 Full Analysis Text"):
            st.text(full_text)
        
        st.caption("🎓 DSBDM Mini Project - TF-IDF + Logistic Regression")

st.balloons()