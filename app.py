import streamlit as st
import pickle
import numpy as np

# Set page configuration
st.set_page_config(page_title="Score Predictor", layout="centered")

# Load the trained model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Header
st.title("Student Performance Predictor")
st.write("Adjust the parameters below to predict the outcome based on the trained KNN Regressor.")
st.markdown("---")

# Input features layout
col1, col2 = st.columns(2)

with col1:
    hours_studied = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
    attendance_percent = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)

with col2:
    sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
    previous_scores = st.number_input("Previous Scores", min_value=0.0, max_value=100.0, value=75.0, step=1.0)

st.markdown("---")

# Prediction button
if st.button("Predict Score", type="primary"):
    # The input array must match the exact order of features during training:
    # ['hours_studied', 'sleep_hours', 'attendance_percent', 'previous_scores']
    features = np.array([[hours_studied, sleep_hours, attendance_percent, previous_scores]])
    
    # Generate prediction
    prediction = model.predict(features)
    
    # Display result
    st.success(f"**Predicted Result:** {prediction[0]:.2f}")
