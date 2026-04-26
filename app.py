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
    # Create FULL DataFrame - Use NaN for missing (imputer fills during transform)
    input_dict = {
        "title": title or np.nan,
        "location": location or np.nan,
        "department": department or np.nan,
        "salary_range": salary_range or np.nan,
        "company_profile": company_profile or np.nan,
        "description": description or np.nan,
        "requirements": requirements or np.nan,
        "benefits": benefits or np.nan,
        "employment_type": employment_type if employment_type != "Missing" else np.nan,
        "required_experience": required_experience if required_experience != "Missing" else np.nan,
        "required_education": required_education if required_education != "Missing" else np.nan,
        "industry": industry or np.nan,
        "function": function or np.nan,
        "job_id": 0,
        "telecommuting": 0,
        "has_company_logo": 0,
        "has_questions": 0
    }
    input_df = pd.DataFrame([input_dict])

    # Transform + Predict
    input_transformed = transformer.transform(input_df)
    prediction = model.predict(input_transformed)[0]

    if prediction == 1:
        st.error("⚠️ Fake Job Posting")
    else:
        st.success("✅ Real Job Posting")