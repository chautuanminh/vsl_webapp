from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()
model = YOLO("v10_m_yolo11.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = model(frame)
    annotated = results[0].plot()  # draw boxes
    _, encoded_img = cv2.imencode(".jpg", annotated)
    return {"image": encoded_img.tobytes()}
