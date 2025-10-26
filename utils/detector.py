# utils/detector.py
from ultralytics import YOLO

model = YOLO("models/v10_m_yolo11.pt")

def detect_frame(frame):
    results = model.predict(frame)
    return results
