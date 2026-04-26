import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re

st.set_page_config(layout="wide")
st.title("🤖 Fake Job Detector Pro")

model = pickle.load(open("model_simple.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# 8 Fields
col1, col2 = st.columns(2)
with col1:
    title = st.text_input("**1. Job Title**")
    description = st.text_area("**2. Description**")
    location = st.text_input("**3. Location**")
with col2:
    company = st.text_input("**4. Company**")
    salary = st.text_input("**5. Salary**")
    emp_type = st.selectbox("**6. Type**", ["Full-time", "Part-time"])
    exp_level = st.selectbox("**7. Experience**", ["Entry", "Mid", "Senior"])
logo = st.checkbox("**8. Logo**")

if st.button("🚀 Predict", type="primary"):
    # BETTER text preprocessing
    def clean_text(text):
        if not text: return ""
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        return text
    
    full_text = f"""
    title {clean_text(title)}
    company {clean_text(company)} 
    location {clean_text(location)}
    salary {clean_text(salary)}
    type {emp_type.lower()}
    experience {exp_level.lower()}
    logo {'yes' if logo else 'no'}
    description {clean_text(description)}
    """
    
    X = vectorizer.transform([full_text])
    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    
    # Results
    col1, col2 = st.columns(2)
    with col1:
        if pred == 1:
            st.error("🔴 **FAKE JOB DETECTED**")
        else:
            st.success("🟢 **REAL JOB**")
    with col2:
        st.metric("Confidence", f"{max(probs)*100:.0f}%")
        st.metric("Fake Prob", f"{probs[1]*100:.0f}%")
    
    st.caption("🎓 TF-IDF + Logistic Regression")

# Balloons ONLY after prediction
if 'predicted' not in st.session_state:
    st.session_state.predicted = False

if st.button("🎉") and st.session_state.predicted:
    st.balloons()