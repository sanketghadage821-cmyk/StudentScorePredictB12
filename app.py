import streamlit as st
import pickle
import numpy as np
import os

# 1. Page Configuration & Aesthetic Initializer
st.set_page_config(
    page_title="AuraPredict AI", 
    page_icon="⚡",
    layout="centered"
)

# 2. Advanced CSS Design System Injection
st.markdown("""
    <style>
    /* Global Background and Typography Setup */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% 50%, #1a1c24 0%, #0f1015 100%) !important;
        font-family: 'Inter', sans-serif !important;
        color: #e2e8f0 !important;
    }
    
    /* Glowing Gradient Header Effect */
    .hero-title {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 0px;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 300;
        margin-bottom: 2rem;
    }

    /* Glassmorphism Frosted Input Panel */
    .glass-panel {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 25px;
    }

    /* Neon Holographic Score Display Box */
    .glow-score-box {
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.1) 0%, rgba(79, 172, 254, 0.1) 100%);
        border: 2px solid #00f2fe;
        border-radius: 20px;
        padding: 35px 20px;
        text-align: center;
        box-shadow: 0 0 25px rgba(0, 242, 254, 0.25), inset 0 0 15px rgba(0, 242, 254, 0.1);
        animation: pulse 3s infinite alternate;
        margin-top: 20px;
    }
    
    /* Interactive Elements Custom Formatting */
    div.stSlider > label, div.stNumberInput > label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    /* Streamlit Button Native Overrides via CSS */
    .stButton>button {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%) !important;
        color: #0f1015 !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        transition: all 0.3s ease-in-out !important;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(0, 242, 254, 0.5) !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Model Loading Core Logic
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'model.pkl')
    if not os.path.exists(model_path):
        st.error("### ❌ Configuration Error: `model.pkl` undetected in root workspace.")
        st.stop()
    with open(model_path, 'rb') as file:
        return pickle.load(file)

model = load_model()

# 4. Interface Header Design Elements
st.markdown('<h1 class="hero-title">AURA PREDICT ENGINE</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">High-fidelity student performance projections powered by local KNN clustering.</p>', unsafe_allow_html=True)

# 5. Visual Form Interface Structure
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown("<h3 style='margin-top:0; color:#ffffff; font-size:1.3rem; font-weight:600;'>📊 Input Metrics Matrix</h3>", unsafe_allow_html=True)

hours_studied = st.slider(
    "📚 Daily Hours Studied", 
    min_value=0.0, max_value=24.0, value=7.0, step=0.5
)

sleep_hours = st.slider(
    "🛌 Target Sleep Hours", 
    min_value=0.0, max_value=24.0, value=8.0, step=0.5
)

attendance_percent = st.number_input(
    "🏫 Institutional Attendance Rate (%)", 
    min_value=0.0, max_value=100.0, value=92.0, step=1.0
)

previous_scores = st.number_input(
    "📈 Baseline Assessment Score", 
    min_value=0.0, max_value=100.0, value=80.0, step=1.0
)
st.markdown('</div>', unsafe_allow_html=True)

# 6. Prediction Calculation Engine & Dynamic Visual Feedback
if st.button("RUN ENGINE DIAGNOSTICS", use_container_width=True):
    features = np.array([[hours_studied, sleep_hours, attendance_percent, previous_scores]])
    prediction = model.predict(features)[0]
    
    # Display Gorgeous Glowing Output Card
    st.markdown(f"""
        <div class="glow-score-box">
            <span style='color: #94a3b8; font-weight: 400; letter-spacing: 2px; font-size: 0.85rem;'>PROJECTED PERFORMANCE TARGET</span>
            <h1 style='color: #ffffff; font-size: 4.2rem; margin: 10px 0 0 0; font-weight: 800; text-shadow: 0 0 10px rgba(255,255,255,0.2);'>{prediction:.1f}</h1>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Elegant, understated performance assessments
    if prediction >= 85:
        st.toast("⚡ Optimal parameters detected! Target indicates elite tier performance.", icon="🌟")
    elif prediction >= 65:
        st.toast("⚡ Stable performance path. Minor localized variable changes can elevate tiers.", icon="📈")
    else:
        st.toast("⚡ Vulnerabilities identified. Immediate adjustment to baseline schedule advised.", icon="⚠️")

# Custom unobtrusive layout divider
st.markdown("<p style='text-align: center; margin-top: 5rem; opacity: 0.25; font-size: 0.8rem;'>AuraPredict Platform • V2.5.0-Cyberpunk</p>", unsafe_allow_html=True)
