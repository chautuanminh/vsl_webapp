import streamlit as st
import requests

# --- CONFIG ---
FASTAPI_URL = "http://localhost:8000"   # change to your server URL if deployed

# --- PAGE SETUP ---
st.set_page_config(page_title="YOLO Realtime Detection", layout="centered")
st.title("🚗 YOLO Real-Time Object Detection")
st.markdown("Run your YOLO model with **FastAPI + Streamlit** integration")

# --- BUTTON CONTROLS ---
if "running" not in st.session_state:
    st.session_state.running = False

col1, col2 = st.columns(2)
with col1:
    start = st.button("▶️ Start Detection")
with col2:
    stop = st.button("⏹️ Stop Detection")

# --- STREAM CONTROL LOGIC ---
if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

# --- LIVE STREAM ---
if st.session_state.running:
    st.markdown("### 📷 Live Stream")
    st.markdown(
        f'<img src="{FASTAPI_URL}/video_feed" width="720" />',
        unsafe_allow_html=True
    )
else:
    st.info("Press **Start Detection** to begin streaming.")
