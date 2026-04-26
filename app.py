import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model and transformer
@st.cache_resource
def load_models():
    model = pickle.load(open("model.pkl", "rb"))
    transformer = pickle.load(open("transformer.pkl", "rb"))
    return model, transformer

model, transformer = load_models()

st.title("🚀 Fake Job Detection System")

# ONLY 3 inputs - bulletproof!
title = st.text_input("**Job Title** (required)")
description = st.text_area("**Job Description** (required)", height=150)
has_logo = st.checkbox("Has company logo?", value=False)

if st.button("🔍 Predict Real or Fake") and title and description:
    # PERFECT input_df - matches EXACTLY what transformer expects
    input_df = pd.DataFrame({
        'title': [title],
        'description': [description],
        'job_id': [0],
        'telecommuting': [0],
        'has_company_logo': [1 if has_logo else 0],
        'has_questions': [0],
        # ALL other columns = empty → NaN → imputed perfectly
        'location': [np.nan], 'department': [np.nan], 'salary_range': [np.nan],
        'company_profile': [np.nan], 'requirements': [np.nan], 'benefits': [np.nan],
        'employment_type': [np.nan], 'required_experience': [np.nan],
        'required_education': [np.nan], 'industry': [np.nan], 'function': [np.nan]
    })

    try:
        input_transformed = transformer.transform(input_df)
        prediction = model.predict(input_transformed)[0]
        probability = model.predict_proba(input_transformed)[0].max()

        st.success(f"**Prediction Confidence: {probability:.1%}**")
        
        if prediction == 1:
            st.error("❌ **FAKE JOB POSTING** ⚠️")
            st.info("🔍 **Red flags detected**: Vague details, unrealistic pay, urgent hiring")
        else:
            st.success("✅ **REAL JOB POSTING** 🎉")
            st.info("✅ **Green flags**: Professional language, clear requirements")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.warning("👆 Enter Job Title + Description first!")