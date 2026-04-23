import streamlit as st
import pickle

# 🔷 Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# 🔷 App title
st.title("Fake Job Detection System")
st.write("Enter job details below:")

# 🔷 User inputs
title = st.text_input("Job Title")
description = st.text_area("Job Description")

# 🔷 Predict button
if st.button("Predict"):

    # Combine inputs
    input_data = [title + " " + description]

    # Convert text → numerical
    input_transformed = vectorizer.transform(input_data)

    # Prediction
    prediction = model.predict(input_transformed)

    # Output
    if prediction[0] == 1:
        st.error("⚠️ Fake Job Posting")
    else:
        st.success("✅ Real Job Posting")