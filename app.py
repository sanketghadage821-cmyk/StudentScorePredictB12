import streamlit as st
import pickle
import numpy as np
import os

# Set page configuration
st.set_page_config(page_title="Score Predictor", layout="centered")

# Load the trained model safely using absolute paths
@st.cache_resource
def load_model():
    # Get the directory where app.py is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Create the full path to model.pkl
    model_path = os.path.join(base_dir, 'model.pkl')
    
    with open(model_path, 'rb') as file:
        return pickle.load(file)

model = load_model()

# Header
st.title("Student Performance Predictor")
st.write("Adjust the parameters below to predict the outcome based on the trained KNN Regressor.")
st.markdown("---")

# Vertical Input layout
st.subheader("Student Data Form")
hours_studied = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
attendance_percent = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
previous_scores = st.number_input("Previous Scores", min_value=0.0, max_value=100.0, value=75.0, step=1.0)

st.markdown("---")

# Prediction button
if st.button("Predict Score", type="primary", use_container_width=True):
    # The input array must match the exact order of features during training
    features = np.array([[hours_studied, sleep_hours, attendance_percent, previous_scores]])
    
    # Generate prediction
    prediction = model.predict(features)
    
    # Display result
    st.success(f"**Predicted Result:** {prediction[0]:.2f}")
