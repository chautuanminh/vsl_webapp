import streamlit as st
import cv2
import requests
import numpy as np

st.title("YOLOv11 Real-Time Object Detection")

BACKEND_URL = "http://127.0.0.1:8000/predict"

# initialize webcam
run = st.checkbox("Start camera")
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)  # default camera

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("No frame captured from camera")
        break

    # encode frame as JPEG for sending
    _, buffer = cv2.imencode('.jpg', frame)
    response = requests.post(
        BACKEND_URL,
        files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
    )

    if response.status_code == 200:
        img_bytes = response.json()["image_bytes"]
        nparr = np.frombuffer(bytes(img_bytes, "latin1"), np.uint8)
        annotated = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        FRAME_WINDOW.image(annotated, channels="BGR")
    else:
        FRAME_WINDOW.image(frame, channels="BGR")

cap.release()
