import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.title("✅ WORKING Fake Job Detector")

model = pickle.load(open("model.pkl", "rb"))
transformer = pickle.load(open("transformer.pkl", "rb"))

# Notebook EXACT column order
COLUMNS = ['job_id', 'title', 'location', 'department', 'salary_range', 
           'company_profile', 'description', 'requirements', 'benefits', 
           'telecommuting', 'has_company_logo', 'has_questions', 
           'employment_type', 'required_experience', 'required_education', 
           'industry', 'function']

title = st.text_input("Title")
description = st.text_area("Description")

if st.button("Predict"):
    # EXACT notebook order + NaN
    row = [0, title, np.nan, np.nan, np.nan, np.nan, description, np.nan, np.nan,
           0, False, 0, np.nan, np.nan, np.nan, np.nan, np.nan]
    
    df = pd.DataFrame([row], columns=COLUMNS)
    
    pred = model.predict(transformer.transform(df))[0]
    
    if pred == 1:
        st.error("🔴 FAKE")
    else:
        st.success("🟢 REAL")