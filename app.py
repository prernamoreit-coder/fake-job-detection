import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")
st.title("🤖 Fake Job Detector")

model = pickle.load(open("model.pkl", "rb"))
transformer = pickle.load(open("transformer.pkl", "rb"))

# 5 SAFE fields only
col1, col2 = st.columns(2)
with col1:
    st.header("📝 Job Info")
    title = st.text_input("Title")
    description = st.text_area("Description", height=120)
with col2:
    st.header("🔧 Flags")
    location = st.text_input("Location")
    company = st.text_input("Company") 
    logo = st.checkbox("Has Logo")

if st.button("🚀 Predict", type="primary"):
    # Only SAFE columns + NaN everywhere else
    df = pd.DataFrame([{
        'title': title or np.nan,
        'description': description or np.nan,
        'location': location or np.nan,
        'company_profile': company or np.nan,
        'job_id': 0, 'telecommuting': 0, 
        'has_company_logo': int(logo), 'has_questions': 0
    }])
    # Fill missing columns with NaN
    for col in ['department', 'salary_range', 'requirements', 'benefits', 
                'employment_type', 'required_experience', 'required_education', 
                'industry', 'function']:
        df[col] = np.nan
    
    st.write("Input OK:", df.shape)
    
    pred = model.predict(transformer.transform(df))[0]
    probs = model.predict_proba(transformer.transform(df))[0]
    
    if pred == 1:
        st.error("🔴 **FAKE**")
    else:
        st.success("🟢 **REAL**")
    st.metric("Confidence", f"{max(probs)*100:.0f}%")