import pytest
from unittest.mock import MagicMock
import sys
import numpy as np
import cv2

# --- MOCKING BEFORE IMPORT ---
# We mock utils.detector so we don't load the heavy YOLO model
mock_detector_module = MagicMock()
sys.modules['utils.detector'] = mock_detector_module

# Setup the Mock Class and Instance
mock_detector_class = MagicMock()
mock_detector_instance = MagicMock()
mock_detector_module.ObjectDetector = mock_detector_class
mock_detector_class.return_value = mock_detector_instance

# Now we can import app
from app import app, detector
from fastapi.testclient import TestClient

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_detect_image():
    # Create a dummy image (100x100 black square)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode('.jpg', img)

    # Configure the mock to return the image and a fake confidence
    # predict_frame returns (annotated_frame, avg_conf)
    detector.predict_frame.return_value = (img, 0.88)

    # Prepare file upload
    files = {'file': ('test.jpg', img_encoded.tobytes(), 'image/jpeg')}

    # Send POST request
    response = client.post("/detect_image", files=files, data={'conf': 0.5})

    assert response.status_code == 200
    json_resp = response.json()

    # Verify response structure
    assert "image" in json_resp
    assert json_resp["image"].startswith("data:image/jpeg;base64,")
    assert "avg_conf" in json_resp
    assert json_resp["avg_conf"] == 0.88
