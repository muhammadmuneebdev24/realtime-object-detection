from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import threading
import time
import os
import shutil

from detector import detect_objects
from tracker import track_objects

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

os.makedirs("uploads", exist_ok=True)

# ── Global state ──────────────────────────────────────────
source_config = {"type": None, "path": None}


# ── Background frame reader ───────────────────────────────
class FrameBuffer:
    def __init__(self):
        self.frame = None
        self._lock = threading.Lock()
        self.running = False
        self._cap = None
        self._frame_delay = 0  # seconds to wait between reads (for video pacing)

    def start(self, source):
        self.running = True
        if isinstance(source, int):
            # Webcam: use DirectShow on Windows for instant start
            self._cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._cap.set(cv2.CAP_PROP_FPS, 30)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._frame_delay = 0  # webcam: read as fast as possible
        else:
            # Video file: pace reads at the video's native FPS
            self._cap = cv2.VideoCapture(source)
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            if self._cap and self._cap.isOpened():
                ret, frame = self._cap.read()
                if ret:
                    with self._lock:
                        self.frame = frame
                    if self._frame_delay > 0:
                        # Pace video to its native FPS so generator keeps up
                        time.sleep(self._frame_delay)
                else:
                    self.running = False
            else:
                time.sleep(0.01)

    def read(self):
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self._cap:
            self._cap.release()
        self._cap = None
        self.frame = None


frame_buffer = FrameBuffer()

PALETTE = [
    (0, 230, 118), (33, 150, 243), (255, 87, 34),
    (156, 39, 176), (255, 193, 7), (0, 188, 212),
    (233, 30, 99), (76, 175, 80), (255, 152, 0),
]


def color_for_id(tid):
    return PALETTE[int(tid) % len(PALETTE)]


# ── MJPEG frame generator ─────────────────────────────────
def generate_frames():
    frame_count = 0
    SKIP = 1          # Detect every frame (tracker is fast now, no embedder)
    last_tracks = []
    fps_start = time.time()
    fps_counter = 0
    current_fps = 0.0

    while frame_buffer.running:
        frame = frame_buffer.read()
        if frame is None:
            time.sleep(0.005)
            continue

        frame_count += 1
        fps_counter += 1

        if fps_counter >= 30:
            elapsed = time.time() - fps_start
            current_fps = fps_counter / elapsed if elapsed > 0 else 0
            fps_start = time.time()
            fps_counter = 0

        if frame_count % SKIP == 0:
            detections = detect_objects(frame)
            last_tracks = track_objects(detections, frame)

        active_count = 0
        for track in last_tracks:
            if not track.is_confirmed():
                continue
            active_count += 1
            tid = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            label = track.get_det_class() or "Object"
            color = color_for_id(tid)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            tag = f"{label} #{tid}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, tag, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        hud = f"FPS: {current_fps:.1f}  Objects: {active_count}"
        cv2.rectangle(frame, (0, 0), (240, 30), (20, 20, 20), -1)
        cv2.putText(frame, hud, (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 118), 1, cv2.LINE_AA)

        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if not ret:
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buffer.tobytes() + b'\r\n')


# ── Routes ────────────────────────────────────────────────

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "streaming": frame_buffer.running,
        "source_type": source_config.get("type", ""),
    })


@app.post("/start_webcam")
async def start_webcam():
    global frame_buffer
    frame_buffer.stop()
    frame_buffer = FrameBuffer()
    source_config["type"] = "webcam"
    frame_buffer.start(0)
    return RedirectResponse(url="/", status_code=303)


@app.post("/start_video")
async def start_video(file: UploadFile = File(...)):
    global frame_buffer
    frame_buffer.stop()
    frame_buffer = FrameBuffer()
    path = f"uploads/{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    source_config["type"] = "video"
    frame_buffer.start(path)
    return RedirectResponse(url="/", status_code=303)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/stop")
def stop_stream():
    frame_buffer.stop()
    return RedirectResponse(url="/", status_code=303)