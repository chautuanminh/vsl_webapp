from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import cv2
import time
import numpy as np

# --- Initialize FastAPI and YOLO model ---
app = FastAPI(title="YOLO Real-Time Detection")
model = YOLO("models/v10_m_yolo11.pt")  # your trained model file

# --- Helper: Draw FPS and Confidence ---
def draw_info(frame, fps, conf):
    text_fps = f"FPS: {fps:.2f}"
    text_conf = f"Avg Conf: {conf:.2f}"
    cv2.putText(frame, text_fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, text_conf, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return frame

# --- Generator for webcam frames ---
def generate_frames():
    cap = cv2.VideoCapture(0)
    prev_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break

        # --- Measure FPS ---
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # --- YOLO inference ---
        results = model(frame)
        annotated_frame = results[0].plot()

        # --- Calculate average confidence ---
        confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else []
        avg_conf = np.mean(confs) if len(confs) > 0 else 0.0

        # --- Draw FPS and confidence on frame ---
        annotated_frame = draw_info(annotated_frame, fps, avg_conf)

        # --- Encode and stream ---
        _, buffer = cv2.imencode(".jpg", annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()

# --- API route for video feed ---
@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# --- Root endpoint ---
@app.get("/")
def root():
    return {"message": "YOLO real-time detection. Visit /video_feed to see the camera stream."}
