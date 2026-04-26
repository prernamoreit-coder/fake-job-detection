import streamlit as st
import pickle
import pandas as pd
import numpy as np

@st.cache_resource
def load_models():
    return pickle.load(open("model.pkl", "rb")), pickle.load(open("transformer.pkl", "rb"))

model, transformer = load_models()

st.title("🚀 Fake Job Detection System - 98% Accurate")

col1, col2 = st.columns([4, 1])
with col1:
    title = st.text_input("**Job Title**", placeholder="e.g. Senior Python Developer")
    description = st.text_area("**Job Description**", height=100, 
                              placeholder="Paste full job posting here...")
with col2:
    has_logo = st.checkbox("Has logo?", value=True)

if st.button("🔍 DETECT FAKE JOB", type="primary") and title and description:
    
    # EXACT 17 columns with NaN for missing (matches notebook preprocessing)
    input_df = pd.DataFrame([{
        'title': title,
        'location': np.nan,
        'department': np.nan,
        'salary_range': np.nan,
        'company_profile': np.nan,
        'description': description,
        'requirements': np.nan,
        'benefits': np.nan,
        'employment_type': np.nan,
        'required_experience': np.nan,
        'required_education': np.nan,
        'industry': np.nan,
        'function': np.nan,
        'telecommuting': 0,
        'has_company_logo': 1 if has_logo else 0,
        'has_questions': 0,
        'job_id': 0
    }])
    
    st.info(f"✅ Input ready: {input_df.shape}")
    
    input_transformed = transformer.transform(input_df)
    prediction = model.predict(input_transformed)[0]
    probs = model.predict_proba(input_transformed)[0]
    confidence = max(probs) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if prediction == 1:
            st.error("🔴 **FAKE JOB POSTING**")
        else:
            st.success("🟢 **LEGITIMATE JOB**")
    with col2:
        st.metric("Confidence", f"{confidence:.0f}%")
    with col3:
        st.metric("Fake Probability", f"{probs[1]*100:.0f}%")
        
    # Key features that mattered
    st.caption(f"💡 Analyzed: title + description + {'logo' if has_logo else 'no logo'}")
    
else:
    st.info("👆 Enter title + description to scan!")
    st.caption("Built for DSBDM Mini Project - DecisionTree 98% accuracy[file:1]")