import streamlit as st
import pickle
import pandas as pd
import numpy as np  # For NaN/None handling

# Load model and transformer
model = pickle.load(open("model.pkl", "rb"))
transformer = pickle.load(open("transformer.pkl", "rb"))

st.title("Fake Job Detection System")

# User inputs (expanded for key features; defaults handle missing)
title = st.text_input("Job Title", value="")
description = st.text_area("Job Description", value="")
location = st.text_input("Location", value="Missing")
department = st.text_input("Department", value="Missing")
salary_range = st.text_input("Salary Range", value="Missing")
company_profile = st.text_area("Company Profile", value="Missing")
requirements = st.text_area("Requirements", value="Missing")
benefits = st.text_area("Benefits", value="Missing")
employment_type = st.selectbox("Employment Type", 
                              ["Full-time", "Other", "Contract", "Temporary", "Part-time"])
required_experience = st.selectbox("Required Experience", 
                                 ["Not Applicable", "Entry level", "Internship", "Mid-Senior level", "Executive", "Associate", "Senior level"])

required_education = st.selectbox("Required Education", 
                                 ["Bachelor's Degree", "High School or equivalent", "Doctorate", "Master's Degree", "Professional", "College Degree", "Vocational", "Certification", "Associate Degree"])
industry = st.text_input("Industry", value="Missing")
function = st.text_input("Function", value="Missing")

if st.button("Predict"):
    # MINIMAL input_df - ONLY columns that ALWAYS work (title + description + numerics)
    input_df = pd.DataFrame({
        'title': [title],
        'description': [description],
        'job_id': [0],
        'telecommuting': [0],
        'has_company_logo': [0],
        'has_questions': [0],
        # Fill ALL other columns with "" (becomes NaN → imputed)
        'location': [''], 'department': [''], 'salary_range': [''],
        'company_profile': [''], 'requirements': [''], 'benefits': [''],
        'employment_type': [''], 'required_experience': [''], 
        'required_education': [''], 'industry': [''], 'function': ['']
    })

    input_transformed = transformer.transform(input_df)
    prediction = model.predict(input_transformed)[0]

    st.write(f"**Raw prediction:** {prediction}")  # Debug
    if prediction == 1:
        st.error("⚠️ Fake Job Posting")
    else:
        st.success("✅ Real Job Posting")