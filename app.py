from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import time
import numpy as np
import io
import base64
from utils.detector import ObjectDetector

# --- Initialize FastAPI and Model ---
app = FastAPI(title="YOLO Web App")
detector = ObjectDetector()

# --- Mount Static Files ---
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Templates ---
templates = Jinja2Templates(directory="templates")

# --- Helper: Draw FPS on frame ---
def draw_fps(frame, fps):
    text_fps = f"FPS: {fps:.2f}"
    cv2.putText(frame, text_fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame

# --- Generator for webcam frames ---
def generate_frames(conf_threshold=0.25):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Fallback for when camera is not available (e.g. CI/CD or no cam)
        # Return a black frame with text
        while True:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No Camera Found", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(1)

    prev_time = time.time()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # --- Measure FPS ---
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time

            # --- Detection ---
            annotated_frame, _ = detector.predict_frame(frame, conf_threshold=conf_threshold)
            annotated_frame = draw_fps(annotated_frame, fps)

            # --- Encode and stream ---
            _, buffer = cv2.imencode(".jpg", annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    finally:
        cap.release()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
def video_feed(conf: float = 0.25):
    return StreamingResponse(
        generate_frames(conf_threshold=conf),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/detect_image")
async def detect_image(file: UploadFile = File(...), conf: float = Form(0.25)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    annotated_img, avg_conf = detector.predict_frame(img, conf_threshold=conf)

    # Encode back to base64 to send to frontend
    _, buffer = cv2.imencode(".jpg", annotated_img)
    img_str = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse(content={
        "image": f"data:image/jpeg;base64,{img_str}",
        "avg_conf": float(avg_conf)
    })
