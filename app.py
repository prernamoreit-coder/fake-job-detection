import streamlit as st
import pickle
import pandas as pd

# Load model and transformer
model = pickle.load(open("model.pkl", "rb"))
transformer = pickle.load(open("transformer.pkl", "rb"))

st.title("Fake Job Detection System")

# User inputs (keep simple)
title = st.text_input("Job Title")
description = st.text_area("Job Description")

if st.button("Predict"):

    # 🔥 Create FULL dataframe with ALL required columns
    input_df = pd.DataFrame([{
    "job_id": 0,
    "title": title,
    "description": description,
    "telecommuting": 0,
    "has_company_logo": 0,
    "has_questions": 0,
    "salary_range": "unknown"
}])

    # 🔥 FORCE correct column order (VERY IMPORTANT)
    expected_columns = [
        "title", "location", "department", "company_profile",
        "description", "requirements", "benefits",
        "employment_type", "required_experience",
        "required_education", "industry", "function"
    ]

    input_df = input_df[expected_columns]

    # Transform + Predict
    input_transformed = transformer.transform(input_df)
    prediction = model.predict(input_transformed)

    if prediction[0] == 1:
        st.error("⚠️ Fake Job Posting")
    else:
        st.success("✅ Real Job Posting")