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
employment_type = st.selectbox("Employment Type", ["Missing", "Full-time", "Other", "Contract", "Temporary"])
required_experience = st.selectbox("Required Experience", ["Missing", "Not Applicable", "Entry level", "Internship", "Mid-Senior level"])
required_education = st.selectbox("Required Education", ["Missing", "Bachelor's Degree", "High School", "Professional"])
industry = st.text_input("Industry", value="Missing")
function = st.text_input("Function", value="Missing")

if st.button("Predict"):
    # Create FULL DataFrame with ALL 17 required columns (match notebook preprocessing)
    input_df = pd.DataFrame({
        "title": [title],
        "location": [location],
        "department": [department],
        "salary_range": [salary_range],  # Matches dataset casing
        "company_profile": [company_profile],
        "description": [description],
        "requirements": [requirements],
        "benefits": [benefits],
        "employment_type": [employment_type],
        "required_experience": [required_experience],
        "required_education": [required_education],
        "industry": [industry],
        "function": [function],
        "job_id": [0],
        "telecommuting": [0],
        "has_company_logo": [0],
        "has_questions": [0]
    })

    # Transform + Predict (no manual reordering needed; ColumnTransformer handles column names)
    input_transformed = transformer.transform(input_df)
    prediction = model.predict(input_transformed)[0]

    if prediction == 1:
        st.error("⚠️ Fake Job Posting")
    else:
        st.success("✅ Real Job Posting")