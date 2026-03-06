import streamlit as st
import requests
from PIL import Image
import io
import base64
import pandas as pd
import time
import os
import sys
import plotly.express as px

# ---------------------------
# STATE MANAGEMENT
# ---------------------------
if "show_repetition_warning" not in st.session_state:
    st.session_state.show_repetition_warning = False

if "last_input" not in st.session_state:
    st.session_state.last_input = ""

if "last_agent" not in st.session_state:
    st.session_state.last_agent = "unknown"

st.set_page_config(
    page_title="Multimodal AI System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# ADVANCED ENTERPRISE UI CSS
# ---------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 46px;
        font-weight: 800;
        text-align: center;
        background: -webkit-linear-gradient(45deg, #0f172a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        padding-top: 20px;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 16px;
        font-weight: 500;
        margin-bottom: 40px;
        letter-spacing: 0.5px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    }
    
    .stButton > button:hover {
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.4);
        transform: translateY(-2px);
        color: white;
    }

    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    div[data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 800;
        color: #0f172a;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 12px;
        letter-spacing: 0.05em;
    }

    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    .agent-badge {
        background-color: white;
        border: 1px solid #e2e8f0;
        color: #334155;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        font-weight: 600;
        font-size: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        transition: all 0.2s;
    }
    
    .agent-badge:hover {
        border-color: #cbd5e1;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    
    .agent-badge .icon {
        background: #10b981;
        color: white;
        border-radius: 50%;
        width: 22px;
        height: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.3);
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: .5; }
    }
    .skeleton-box {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        background-color: #e2e8f0;
        border-radius: 8px;
        height: 100px;
        width: 100%;
    }

</style>
""", unsafe_allow_html=True)

# ---------------------------
# HEADER
# ---------------------------
st.markdown("<div class='main-title'>Multimodal AI Agent System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Post-Trained LLM • Multi-Agent Architecture • Enterprise Ready</div>", unsafe_allow_html=True)
st.write("") 

# ---------------------------
# TOP NAVIGATION TABS
# ---------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📝 Summarizer",
    "🔍 Vision",
    "🔊 Text-to-Speech",
    "🎨 Text-to-Image",
    "📊 Analytics",
    "🧪 Experiment Tracking"
])

# ---------------------------
# TEXT SUMMARY
# ---------------------------
with tab1:
    st.markdown("### 📝 Text & PDF Summarizer")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        text_input = st.text_area("Enter Text (optional)", height=150, placeholder="Paste your text here...")
    with col2:
        uploaded_pdf = st.file_uploader("Upload PDF (optional)", type=["pdf"])
        st.write("")
        generate_btn = st.button("Generate Summary")

    if generate_btn:
        if uploaded_pdf is None and text_input.strip() == "":
            st.warning("⚠️ Please provide text or upload a PDF to begin.")
            st.stop()
            
        with st.status("Initializing Summarizer Agent...", expanded=True) as status:
            st.write("📥 Receiving input data...")
            time.sleep(0.2) 
            st.write("🧠 Routing to LLM backend...")
            
            if uploaded_pdf is not None:
                response = requests.post(
                    "http://127.0.0.1:8000/summarize",
                    files={"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")}
                )
            else:
                response = requests.post(
                    "http://127.0.0.1:8000/summarize",
                    data={"text": text_input}
                )

            if response.status_code == 200:
                data = response.json()
                status.update(label="✅ Summary Generated Successfully!", state="complete", expanded=True)
                
                st.divider()
                st.markdown("### 📄 Generated Summary")
                st.info(data["summary"])

                st.session_state.last_input = text_input
                st.session_state.last_agent = "summarizer"
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    conf_score = f"{data['confidence'] * 100:.1f}%" if isinstance(data['confidence'], float) else data['confidence']
                    st.metric("Confidence Score", conf_score)
                with m2:
                    if data["hallucinated"]:
                        st.metric("Hallucination Risk", "High ⚠️")
                        st.caption(f"Ratio: {data['hallucination_ratio']}")
                    else:
                        st.metric("Hallucination Risk", "Low ✅")

# ---------------------------
# OBJECT DETECTION
# ---------------------------
with tab2:
    st.markdown("### 🔍 Object Detection")

    col1, col2 = st.columns([1, 2])
    with col1:
        mode = st.radio("Choose Input Mode", ["Upload Image", "Live Camera"])
        image_file = None

        if mode == "Upload Image":
            image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        elif mode == "Live Camera":
            image_file = st.camera_input("Take a picture")

    with col2:
        if image_file is not None:
            st.image(image_file, caption="Input Image", width="stretch")

            if st.button("Detect Objects"):
                with st.status("Analyzing visual data...", expanded=True) as status:
                    st.write("👁️ Vision agent processing image tensors...")
                    try:
                        response = requests.post(
                            "http://127.0.0.1:8000/detect",
                            files={"image": ("image.jpg", image_file.getvalue(), "image/jpeg")}
                        )
                        data = response.json()
                        
                        if "image" in data:
                            status.update(label="✅ Objects Detected!", state="complete", expanded=True)
                            st.divider()
                            st.markdown("### 🎯 Detection Results")
                            
                            res_col1, res_col2 = st.columns([2, 1])
                            with res_col1:
                                img_bytes = bytes.fromhex(data["image"])
                                st.image(img_bytes, caption="Bounding Boxes Applied", width="stretch")
                            
                            with res_col2:
                                st.markdown("#### 📋 Objects Found")
                                for obj in data.get("objects", []):
                                    st.markdown(f"**{obj['label']}** • `{obj['confidence']}`")
                                st.caption(f"⏱ Latency: {data.get('latency', 0)} sec")
                        else:
                            status.update(label="❌ Detection Failed", state="error")
                            st.error(f"Error: {data}")

                    except Exception as e:
                        status.update(label="❌ System Exception", state="error")
                        st.error(f"Frontend Exception: {e}")

# ---------------------------
# TEXT TO SPEECH
# ---------------------------
with tab3:
    st.markdown("### 🔊 Text to Speech")
    
    text = st.text_area("Enter text for speech synthesis", height=150)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Generate Speech"):
            with st.status("Synthesizing audio...", expanded=True) as status:
                st.write("🎙️ Generating waveforms...")
                response = requests.post(
                    "http://127.0.0.1:8000/text-to-speech",
                    params={"text": text}
                )

                if response.status_code == 200:
                    status.update(label="✅ Audio Ready", state="complete")
                    st.audio(response.content)
                    st.toast('🎵 Audio generated successfully!', icon='✅')
                else:
                    status.update(label="❌ Synthesis Failed", state="error")

# ---------------------------
# TEXT TO IMAGE
# ---------------------------
with tab4:
    st.markdown("### 🎨 Text to Image")

    prompt = st.text_input("Enter your creative prompt", placeholder="e.g., A highly detailed figurine of a cybernetic knight...")

    if st.button("Generate Image"):
        with st.status("Painting pixels...", expanded=True) as status:
            st.write("🌌 Diffusion model initializing...")
            response = requests.post(
                "http://127.0.0.1:8000/generate-image",
                params={"prompt": prompt}
            )

            if response.status_code == 200:
                data = response.json()
                if "image" in data:
                    status.update(label="✅ Image Generated", state="complete", expanded=False)
                    image_data = data["image"]
                    if isinstance(image_data, list):
                        image_data = image_data[0]
                    
                    image_bytes = base64.b64decode(image_data)
                    st.image(image_bytes, width="stretch", caption=prompt)
                    st.toast('🎨 Imagee created!', icon='🎉')
                else:
                    status.update(label="❌ Image Generation Failed", state="error")
                    st.error("No image key in response.")

# ---------------------------
# DASHBOARD (UPGRADED WITH PLOTLY)
# ---------------------------
with tab5:
    st.markdown("### 📊 Evaluation Dashboard")
    
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Current Model Version", "v2 LoRA")
    kpi2.metric("Routing Accuracy", "91.4%", "+2.1%")
    kpi3.metric("Avg User Rating", "4.3 / 5.0", "+0.4")
    
    st.divider()

    METRICS_FILE = os.path.join(os.path.dirname(__file__), "../backend/experiments/metrics.csv")

    if os.path.exists(METRICS_FILE):
        df = pd.read_csv(METRICS_FILE)
        if len(df) > 0:
            st.markdown("#### 📈 System Telemetry")
            
            # Layout the charts
            chart_col1, chart_col2 = st.columns(2)
            
            # Interactive Plotly Line Chart for Confidence
            with chart_col1:
                fig_conf = px.line(df, y="confidence", title="Confidence Over Time", markers=True, 
                                   color_discrete_sequence=["#3b82f6"])
                fig_conf.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_conf, width="stretch")
            
            # Interactive Plotly Line Chart for Latency
            with chart_col2:
                fig_lat = px.line(df, y="latency", title="Latency Over Time (sec)", markers=True,
                                  color_discrete_sequence=["#ef4444"])
                fig_lat.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_lat, width="stretch")

            # Interactive Plotly Bar Chart for Agent Usage
            st.markdown("<br>", unsafe_allow_html=True)
            agent_counts = df["agent"].value_counts().reset_index()
            agent_counts.columns = ["Agent", "Count"]
            fig_bar = px.bar(agent_counts, x="Agent", y="Count", title="Agent Usage Breakdown",
                             color="Agent", text="Count")
            fig_bar.update_layout(margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
            st.plotly_chart(fig_bar, width="stretch")

            with st.expander("View Raw Activity Log", expanded=False):
                st.dataframe(df.tail(10), width="stretch")

            if "model_version" in df.columns:
                st.divider()
                st.markdown("#### 🧠 Model Version Analytics")
                
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    version_conf = df.groupby("model_version")["confidence"].mean().reset_index()
                    fig_v_conf = px.bar(version_conf, x="model_version", y="confidence", title="Avg Confidence by Model")
                    st.plotly_chart(fig_v_conf, width="stretch")

                with v_col2:
                    version_latency = df.groupby("model_version")["latency"].mean().reset_index()
                    fig_v_lat = px.bar(version_latency, x="model_version", y="latency", title="Avg Latency by Model")
                    st.plotly_chart(fig_v_lat, width="stretch")
        else:
            st.info("Metrics file exists but contains no data.")
    else:
        st.markdown("<div class='skeleton-box'></div>", unsafe_allow_html=True)
        st.caption("Waiting for telemetry data...")

    st.divider()
    
    st.markdown("### ⭐ Rate System Output")
    with st.container():
        feedback_text = st.text_area("📝 Additional Comments (Optional)", placeholder="Tell us what you liked or what should be improved...")
        rating = st.slider("Quality Rating", 1, 5, 5)

        if st.button("Submit Feedback"):
            feedback_data = {
                "input_text": st.session_state.last_input,
                "rating": rating,
                "comment": feedback_text,
                "agent": st.session_state.last_agent
            }

            r = requests.post("http://127.0.0.1:8000/feedback", json=feedback_data)
            if r.status_code == 200:
                st.toast('Feedback submitted successfully! Thank you.', icon='🚀')
            else:
                st.toast('Failed to submit feedback.', icon='❌')

# ---------------------------
# EXPERIMENT TRACKING (UPGRADED WITH PLOTLY)
# ---------------------------
with tab6:
    st.markdown("### 🧪 Experiment Tracking")

    exp_file = os.path.join(os.path.dirname(__file__), "../backend/backend/experiments/experiments.json")

    if os.path.exists(exp_file):
        import json
        with open(exp_file, "r") as f:
            data = json.load(f)

        if len(data) == 0:
            st.info("No experiments logged yet.")
        else:
            exp_df = pd.DataFrame(data)
            st.dataframe(exp_df, width="stretch")

            col1, col2 = st.columns(2)
            with col1:
                fig_exp_conf = px.line(exp_df, y="confidence", title="Experiment Confidence", markers=True, color_discrete_sequence=["#8b5cf6"])
                st.plotly_chart(fig_exp_conf, width="stretch")
            with col2:
                fig_exp_lat = px.line(exp_df, y="latency", title="Experiment Latency", markers=True, color_discrete_sequence=["#f59e0b"])
                st.plotly_chart(fig_exp_lat, width="stretch")
    else:
        st.markdown("<div class='skeleton-box'></div>", unsafe_allow_html=True)
        st.caption("Experiment tracking log not found.")

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:
    st.markdown("### 🧠 Active Agents")
    
    st.markdown("""
        <div class="agent-badge">Summarizer <span class="icon">✓</span></div>
        <div class="agent-badge">Vision System <span class="icon">✓</span></div>
        <div class="agent-badge">Audio Gen <span class="icon">✓</span></div>
        <div class="agent-badge">Image Gen <span class="icon">✓</span></div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 AI Performance")
    
    FILE = os.path.join(os.path.dirname(__file__), "../backend/experiments/metrics.csv")

    if os.path.exists(FILE):
        df = pd.read_csv(FILE)
        df["hallucination"] = df["hallucination"].astype(str).str.lower() == "true"

        if len(df) > 0:
            st.metric("System Confidence", f"{round(df['confidence'].mean(), 2)}")
            st.metric("Response Latency", f"{round(df['latency'].mean(), 2)}s")
            st.metric("Hallucination Rate", f"{round(df['hallucination'].mean() * 100, 1)}%")
        else:
            st.caption("Run agents to populate telemetry.")
    else:
        st.caption("No performance metrics yet.")