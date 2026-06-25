# Object Detection and Tracking

This is my Task 4 project for real-time object detection and tracking using YOLOv8 and a custom IoU tracker. It runs in the browser using FastAPI and OpenCV.

You can use your webcam or upload a video file. It will detect objects and draw bounding boxes with tracking IDs on each frame.

## What it does

- detects objects in real time using YOLOv8n
- tracks each object across frames using IoU matching
- shows live FPS and object count on the video
- supports webcam and video file input
- runs in the browser (no frontend framework needed)

## Requirements

- Python 3.10 or higher
- A webcam (for live mode)

## How to run

1. Clone the repo
```
git clone https://github.com/yourusername/Object_Detection.git
cd Object_Detection
```

2. Create a virtual environment
```
python -m venv venv
```

3. Activate it

On Windows:
```
venv\Scripts\activate
```

On Mac/Linux:
```
source venv/bin/activate
```

4. Install requirements
```
pip install -r requirements.txt
```

5. Run the app
```
uvicorn main:app --reload
```

6. Open your browser and go to `http://127.0.0.1:8000`

The YOLOv8 model will download automatically on first run (about 6MB).

## Project structure

```
Object_Detection/
├── main.py          # FastAPI app, video stream logic
├── detector.py      # YOLOv8 object detection
├── tracker.py       # Simple IoU-based tracker
├── templates/
│   └── index.html   # Main webpage
├── static/
│   └── style.css    # Basic styles
├── uploads/         # Uploaded videos stored here
└── requirements.txt
```

## Libraries used

- [FastAPI](https://fastapi.tiangolo.com/) - web framework
- [Uvicorn](https://www.uvicorn.org/) - ASGI server
- [OpenCV](https://opencv.org/) - video capture and drawing
- [Ultralytics YOLOv8](https://docs.ultralytics.com/) - object detection model
- [Jinja2](https://jinja.palletsprojects.com/) - HTML templates

## Notes

- The app runs on CPU so FPS depends on your machine. On a decent laptop you should get around 10-15 FPS.
- The tracker uses IoU overlap to match objects between frames, so it works without any extra neural network.
- I built a custom tracker instead of Deep SORT because Deep SORT was too slow on CPU.
