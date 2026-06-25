from ultralytics import YOLO
from pathlib import Path

# Load from Ultralytics cache — model stays in venv/user cache, NOT the project folder.
# YOLO will auto-download here on first run if missing.
MODEL_PATH = Path.home() / "AppData" / "Roaming" / "Ultralytics" / "yolov8n.pt"
model = YOLO(str(MODEL_PATH))


def detect_objects(frame):
    results = model.predict(frame, imgsz=256, conf=0.3, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = model.names[int(box.cls[0])]
            detections.append([x1, y1, x2, y2, conf, label])
    return detections