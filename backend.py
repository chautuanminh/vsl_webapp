from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()
model = YOLO("v10_m_yolo11.pt")

# Allow Streamlit (or any frontend) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run YOLO inference
    results = model(frame)
    annotated = results[0].plot()

    # Encode annotated image to JPEG
    success, buffer = cv2.imencode(".jpg", annotated)
    if not success:
        return {"error": "Failed to encode image"}

    # Return as an actual image response
    return Response(content=buffer.tobytes(), media_type="image/jpeg")
