import streamlit as st
import requests
import cv2
import numpy as np

st.title("Real-time Object Detection")
camera = st.camera_input("Take a picture")

if camera:
    files = {"file": camera.getvalue()}
    response = requests.post("http://127.0.0.1:8000/predict", files=files)
    if response.status_code == 200:
        img_bytes = response.content
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        st.image(img, channels="BGR")
