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

col1, col2 = st.columns([3,1])
title = col1.text_input("**Job Title**", key="title")
description = col1.text_area("**Job Description**", height=120, key="desc")
has_logo = col2.checkbox("✅ Company Logo?", value=False)

if st.button("🔍 Predict", type="primary") and title.strip() and description.strip():
    
    # EXACT column order + types from your notebook training data
    input_data = {
        'title': title,
        'location': '',
        'department': '',
        'salary_range': '',
        'company_profile': '',
        'description': description,
        'requirements': '',
        'benefits': '',
        'telecommuting': 0,
        'has_company_logo': 1 if has_logo else 0,
        'has_questions': 0,
        'employment_type': '',
        'required_experience': '',
        'required_education': '',
        'industry': '',
        'function': '',
        'job_id': 0
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Ensure correct dtypes (matches training)
    for col in input_df.columns:
        if col in ['telecommuting', 'has_company_logo', 'has_questions', 'job_id']:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0).astype(int)
        else:
            input_df[col] = input_df[col].astype(str)
    
    st.info(f"📊 Input shape: {input_df.shape}")
    
    try:
        input_transformed = transformer.transform(input_df)
        prediction = model.predict(input_transformed)[0]
        prob = model.predict_proba(input_transformed)[0]
        confidence = max(prob) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.error("❌ **FAKE JOB** ⚠️")
            else:
                st.success("✅ **REAL JOB** 🎉")
        with col2:
            st.metric("Confidence", f"{confidence:.0f}%")
            
        st.caption("💡 Powered by Decision Tree (98% accurate on test set)")
        
    except Exception as e:
        st.error(f"❌ Transform error: {str(e)[:200]}...")
        st.code(str(input_df.dtypes))
else:
    st.warning("👆 Fill title + description")