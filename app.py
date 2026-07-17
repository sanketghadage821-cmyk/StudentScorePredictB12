import streamlit as st
import pickle
import numpy as np
import os

# 1. Advanced Page Configuration
st.set_page_config(
    page_title="EduPredict Engine", 
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Injection
st.markdown("""
    <style>
    /* Main container adjustments */
    .reportview-container {
        background: #1e222b;
    }
    /* Custom Card Design */
    .metric-card {
        background-color: #2b303c;
        border-left: 5px solid #00ADB5;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .result-card {
        background: linear-gradient(135deg, #1f4068, #162447);
        border: 1px solid #00ADB5;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    /* Input adjustments */
    div.stNumberInput > label {
        font-weight: 600 !important;
        color: #EEEEEE !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Robust Model Loader with Debugging UI
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'model.pkl')
    
    if not os.path.exists(model_path):
        st.error("### ❌ Model File Missing")
        st.info(f"**Directory Scanned:** `{base_dir}` \n\n**Detected Files:** {os.listdir(base_dir)}")
        st.warning("Please push `model.pkl` to your GitHub repo root folder.")
        st.stop()
        
    with open(model_path, 'rb') as file:
        return pickle.load(file)

model = load_model()

# 3. Sidebar - Analytics & Metadata Documentation
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/000000/artificial-intelligence.png", width=80)
    st.title("Model Intelligence")
    st.markdown("---")
    st.markdown("### 🧠 Engine Specifications")
    st.write("**Algorithm:** K-Nearest Neighbors Regressor")
    st.write("**Environment:** Scikit-Learn v1.6.1")
    st.markdown("---")
    st.markdown("### 📋 Input Features Requirements")
    st.markdown("""
    * **Hours Studied:** Daily preparation scale.
    * **Sleep Hours:** Rest cycle quality.
    * **Attendance:** Classroom engagement metric.
    * **Previous Scores:** Legacy academic baseline.
    """)
    st.caption("© 2026 EduPredict Analytics Engine Engine V2.1")

# 4. Main App Header
st.markdown("<h1 style='text-align: center; color: #00ADB5;'>🎓 EduPredict AI Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.8;'>Advanced predictive analytics modeling student performance outcomes.</p>", unsafe_allow_html=True)
st.markdown("---")

# 5. Tabbed Interface Layout
tab1, tab2 = st.tabs(["🔮 Performance Predictor", "📊 Insights & Metrics"])

with tab1:
    st.markdown("### ✏️ Student Feature Input Form")
    st.write("Provide the student metrics below to evaluate expected testing performance.")
    
    # Custom Card Container for inputs
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    
    hours_studied = st.slider(
        "📚 Daily Hours Studied", 
        min_value=0.0, max_value=24.0, value=6.5, step=0.5,
        help="Average number of hours dedicated to study per day."
    )
    
    sleep_hours = st.slider(
        "🛌 Average Sleep Hours", 
        min_value=0.0, max_value=24.0, value=7.5, step=0.5,
        help="Average hours of sleep the student gets per night."
    )
    
    attendance_percent = st.number_input(
        "🏫 Attendance Rate (%)", 
        min_value=0.0, max_value=100.0, value=90.0, step=1.0,
        help="Overall institutional presence percentage."
    )
    
    previous_scores = st.number_input(
        "📈 Previous Assessment Score", 
        min_value=0.0, max_value=100.0, value=78.0, step=1.0,
        help="The numerical grade achieved in the preceding academic evaluation."
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction Evaluation Execution
    if st.button("🚀 Compute Performance Metrics", type="primary", use_container_width=True):
        
        # Formulate features vector matching exact model expectations
        features = np.array([[hours_studied, sleep_hours, attendance_percent, previous_scores]])
        
        # Inference Generation
        prediction = model.predict(features)[0]
        
        st.markdown("---")
        
        # UI Presentation Card for Predictions
        st.markdown(f"""
            <div class="result-card">
                <h3 style='color: #EEEEEE; margin-bottom: 5px;'>PREDICTED PERFORMANCE SCORE</h3>
                <h1 style='color: #00ADB5; font-size: 3.5rem; margin: 0;'>{prediction:.2f}</h1>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Intelligent, context-aware analytics feedback
        if prediction >= 85:
            st.success("🌟 **Academic Excellence Status:** The current parameters closely correlate with exceptional outcomes. Maintain current study patterns.")
        elif prediction >= 65:
            st.info("👍 **Stable Growth Status:** The parameters indicate acceptable performance. Minor adjustments in attendance or sleep can unlock higher performance tiers.")
        else:
            st.warning("⚠️ **Risk Intervention Required:** The metrics indicate potential academic vulnerabilities. Prioritize increasing attendance and structured daily study blocks.")

with tab2:
    st.markdown("### 📊 Under the Hood: Mathematical Approach")
    st.write("This engine leverages a non-parametric instance-based learning model.")
    
    st.markdown("#### Distance Metrics Calculation")
    st.write("The model classifies and computes projections via Euclidean metrics across multiple dimensions:")
    
    # Utilizing LaTeX blocks purely for scientific formula representations
    st.latex(r"d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}")
    
    st.info("""
    **Analytical Notes:** 
    Because KNN determines predictions based on local feature clustering, extreme changes in inputs (like 24 hours of studying paired with 0 sleep) may default to boundaries calculated from closest historic training parameters.
    """)
