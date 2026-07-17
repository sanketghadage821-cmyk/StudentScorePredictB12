import streamlit as st
import pickle
import numpy as np
import os

# Set page configuration
st.set_page_config(page_title="Score Predictor", layout="centered")

# Load the trained model safely
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'model.pkl')
    
    # Debugging check if the file doesn't exist
    if not os.path.exists(model_path):
        st.error("### ❌ File Not Found Error")
        st.write(f"The app looked for `model.pkl` at: `{model_path}` but couldn't find it.")
        
        # List files to see what actually uploaded
        files_in_repo = os.listdir(base_dir)
        st.info(f"**Current files detected in your GitHub folder:** {files_in_repo}")
        st.warning("Please make sure `model.pkl` is spelled correctly (all lowercase) and successfully pushed to GitHub.")
        st.stop()
        
    with open(model_path, 'rb') as file:
        return pickle.load(file)

# Call the model loading function
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
    features = np.array([[hours_studied, sleep_hours, attendance_percent, previous_scores]])
    prediction = model.predict(features)
    st.success(f"**Predicted Result:** {prediction[0]:.2f}")
