import streamlit as st
import pickle
import pandas as pd

# Load model and transformer
model = pickle.load(open("model.pkl", "rb"))
transformer = pickle.load(open("transformer.pkl", "rb"))

st.title("Fake Job Detection System")

title = st.text_input("Job Title")
description = st.text_area("Job Description")

if st.button("Predict"):

    # Create dataframe (IMPORTANT for ColumnTransformer)
    input_df = pd.DataFrame({
    "title": [title],
    "location": [""],
    "department": [""],
    "company_profile": [""],
    "description": [description],
    "requirements": [""],
    "benefits": [""],
    "employment_type": [""],
    "required_experience": [""],
    "required_education": [""],
    "industry": [""],
    "function": [""]
})

    # Transform using SAME transformer
    input_transformed = transformer.transform(input_df)

    prediction = model.predict(input_transformed)

    if prediction[0] == 1:
        st.error("⚠️ Fake Job Posting")
    else:
        st.success("✅ Real Job Posting")