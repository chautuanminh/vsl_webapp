from ultralytics import YOLO
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, model_path="models/v10_m_yolo11.pt"):
        self.model = YOLO(model_path)

    def predict_frame(self, frame, conf_threshold=0.25):
        # Run inference
        results = self.model(frame, conf=conf_threshold)

        # Plot results on the frame
        annotated_frame = results[0].plot()

        # Calculate average confidence
        confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else []
        avg_conf = np.mean(confs) if len(confs) > 0 else 0.0

        return annotated_frame, avg_conf
