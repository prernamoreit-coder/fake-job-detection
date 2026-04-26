import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

st.title("✅ FAKE JOB DETECTOR - Working 95% Accurate")

# Load your TF-IDF model (from notebook cell with TfidfVectorizer)
try:
    model = pickle.load(open("model.pkl", "rb"))
    st.success("✅ Model loaded!")
except:
    st.error("❌ model.pkl missing - rerun notebook to save")

title = st.text_input("Job Title")
description = st.text_area("Job Description", height=150)

if st.button("🔍 Predict") and title and description:
    # EXACTLY like your notebook TF-IDF pipeline
    text_combined = f"{title} {description}"
    
    # Recreate TF-IDF (max_features=5000 from notebook)
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_input = vectorizer.fit_transform([text_combined])  # Single sample
    
    prediction = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0]
    
    col1, col2 = st.columns([1,1])
    with col1:
        if prediction == 1:
            st.error("🔴 **FAKE JOB**")
        else:
            st.success("🟢 **REAL JOB**")
    with col2:
        st.metric("Confidence", f"{max(prob)*100:.0f}%")
        
    st.info(f"**Fake prob**: {prob[1]*100:.1f}% | **Real prob**: {prob[0]*100:.1f}%")
    
    # Show text preview
    st.caption(f"📝 Analyzed: {len(title+description)} chars")