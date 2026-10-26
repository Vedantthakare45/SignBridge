from flask import Flask, render_template, jsonify, Response, request
import cv2
import mediapipe as mp
import threading
import time
import math
from pathlib import Path
from urllib.request import urlopen

app = Flask(__name__)

# ============================================================
# MediaPipe CURRENT TASKS API
# Works with current MediaPipe releases, including Python 3.13.
# We do NOT use mp.solutions.hands.
# ============================================================

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).parent / "assets" / "hand_landmarker.task"

MESSAGES = {
    "hello": {"en": "Hello", "hi": "नमस्ते", "mr": "नमस्कार"},
    "yes": {"en": "Yes", "hi": "हाँ", "mr": "हो"},
    "fist": {"en": "I need help", "hi": "मुझे मदद चाहिए", "mr": "मला मदत हवी आहे"},
    "peace": {"en": "Peace", "hi": "शांति", "mr": "शांतता"},
    "point": {"en": "Please look here", "hi": "कृपया यहाँ देखें", "mr": "कृपया इथे पहा"},
    "ok": {"en": "Okay", "hi": "ठीक है", "mr": "ठीक आहे"},
}

camera = None
latest_frame = None
camera_lock = threading.Lock()
camera_running = False
camera_thread = None

current_mode = "alphabet"
current_detection = {
    "value": "",
    "meaning": "",
    "confidence": "Waiting"
}

landmarker = None


def ensure_model():
    """Download the 7.8 MB hand-landmarker model once if it is missing."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Hand model not found.")
    print("Downloading hand_landmarker.task (about 8 MB)...")
    print("This is a model file, NOT a Python installation.")

    try:
        with urlopen(MODEL_URL, timeout=60) as response:
            data = response.read()

        if len(data) < 1_000_000:
            raise RuntimeError("Downloaded model file is too small.")

        MODEL_PATH.write_bytes(data)
        print("Hand model downloaded successfully.")
    except Exception as exc:
        raise RuntimeError(
            "\nCould not download hand_landmarker.task.\n"
            "Download it manually from the official MediaPipe model URL:\n"
            f"{MODEL_URL}\n\n"
            f"Then put the file here:\n{MODEL_PATH}\n\n"
            f"Original error: {exc}"
        ) from exc


def create_landmarker():
    global landmarker

    ensure_model()

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )

    landmarker = HandLandmarker.create_from_options(options)


def distance(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2 +
        ((a.z or 0) - (b.z or 0)) ** 2
    )


def finger_state(lm):
    wrist = lm[0]

    return {
        "thumb": distance(lm[4], wrist) > distance(lm[3], wrist) * 1.10,
        "index": distance(lm[8], wrist) > distance(lm[6], wrist) * 1.18,
        "middle": distance(lm[12], wrist) > distance(lm[10], wrist) * 1.18,
        "ring": distance(lm[16], wrist) > distance(lm[14], wrist) * 1.18,
        "pinky": distance(lm[20], wrist) > distance(lm[18], wrist) * 1.18,
    }


def demo_alphabet(lm):
    """
    Prototype alphabet classifier.
    It demonstrates the complete camera -> landmarks -> classifier pipeline.
    A real A-Z ISL model should later replace this rule-based function.
    """
    f = finger_state(lm)
    gap = distance(lm[4], lm[8])
    palm = max(distance(lm[0], lm[9]), 0.001)

    if gap < palm * 0.35 and f["middle"] and f["ring"] and f["pinky"]:
        return "O"

    if not any(f.values()):
        return "A"

    if f["thumb"] and not f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]:
        return "L"

    if f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]:
        return "D"

    if f["index"] and f["middle"] and not f["ring"] and not f["pinky"]:
        return "V"

    if f["index"] and f["middle"] and f["ring"] and not f["pinky"]:
        return "W"

    if f["index"] and f["middle"] and f["ring"] and f["pinky"] and not f["thumb"]:
        return "B"

    if all(f.values()):
        return "Y"

    return None


def demo_word(lm):
    f = finger_state(lm)
    gap = distance(lm[4], lm[8])
    palm = max(distance(lm[0], lm[9]), 0.001)

    if gap < palm * 0.35 and f["middle"] and f["ring"] and f["pinky"]:
        return "ok"

    if all(f.values()):
        return "hello"

    if f["thumb"] and not f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]:
        return "yes"

    if f["index"] and f["middle"] and not f["ring"] and not f["pinky"] and not f["thumb"]:
        return "peace"

    if f["index"] and not f["middle"] and not f["ring"] and not f["pinky"] and not f["thumb"]:
        return "point"

    if not any(f.values()):
        return "fist"

    return None


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    points = []

    for p in landmarks:
        x = max(0, min(w - 1, int(p.x * w)))
        y = max(0, min(h - 1, int(p.y * h)))
        points.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 120), -1)

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 220, 255), 2)


def process_frame(frame, timestamp_ms):
    global current_detection

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    if not result.hand_landmarks:
        current_detection = {
            "value": "",
            "meaning": "",
            "confidence": "No hand"
        }
        return frame

    landmarks = result.hand_landmarks[0]
    draw_landmarks(frame, landmarks)

    if current_mode == "alphabet":
        value = demo_alphabet(landmarks)

        current_detection = {
            "value": value or "",
            "meaning": f"Letter {value}" if value else "",
            "confidence": "Detected" if value else "Hand detected"
        }

    else:
        value = demo_word(landmarks)

        current_detection = {
            "value": value or "",
            "meaning": MESSAGES.get(value, {}).get("en", "") if value else "",
            "confidence": "Detected" if value else "Hand detected"
        }

    return frame


def camera_loop():
    global camera, camera_running, latest_frame

    start_time = time.monotonic()

    while camera_running:
        with camera_lock:
            cap = camera

        if cap is None:
            break

        ok, frame = cap.read()

        if not ok:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)

        timestamp_ms = int(
            (time.monotonic() - start_time) * 1000
        )

        try:
            frame = process_frame(frame, timestamp_ms)
        except Exception as exc:
            current_detection["confidence"] = "Detection error"
            cv2.putText(
                frame,
                "Detection error - see terminal",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            print("Detection error:", exc)

        cv2.putText(
            frame,
            f"Mode: {current_mode.upper()}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        with camera_lock:
            latest_frame = frame

        time.sleep(0.01)


def start_camera():
    global camera, camera_running, camera_thread, latest_frame

    if camera_running:
        return True, "Camera already running"

    latest_frame = None

    if landmarker is None:
        create_landmarker()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        cap.release()
        return False, (
            "Python could not open the webcam. "
            "Check Windows Camera permissions and close Zoom/Teams/OBS."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    camera = cap
    camera_running = True

    camera_thread = threading.Thread(
        target=camera_loop,
        daemon=True
    )
    camera_thread.start()

    return True, "Camera started"


def stop_camera():
    global camera, camera_running, latest_frame

    camera_running = False

    with camera_lock:
        cap = camera
        camera = None
        latest_frame = None

    if cap is not None:
        cap.release()


def generate_frames():
    while camera_running:
        with camera_lock:
            if latest_frame is None:
                frame = None
            else:
                frame = latest_frame.copy()

        if frame is None:
            time.sleep(0.05)
            continue

        ok, buffer = cv2.imencode(".jpg", frame)

        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/detection")
def api_detection():
    return jsonify(current_detection)


@app.post("/api/camera/start")
def api_start_camera():
    try:
        ok, message = start_camera()
        return jsonify({"ok": ok, "message": message}), 200 if ok else 500
    except Exception as exc:
        print("Camera startup error:", exc)
        return jsonify({
            "ok": False,
            "message": str(exc)
        }), 500


@app.post("/api/camera/stop")
def api_stop_camera():
    stop_camera()
    return jsonify({"ok": True, "message": "Camera stopped"})


@app.post("/api/mode")
def api_mode():
    global current_mode

    data = request.get_json(silent=True) or {}
    selected = data.get("mode", "alphabet")

    if selected not in ("alphabet", "word"):
        selected = "alphabet"

    current_mode = selected

    return jsonify({
        "ok": True,
        "mode": current_mode
    })


@app.get("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/health")
def health():
    return jsonify({
        "python_mediapipe": mp.__version__,
        "new_tasks_api": hasattr(mp, "tasks"),
        "model_exists": MODEL_PATH.exists(),
        "camera_running": camera_running
    })


if __name__ == "__main__":
    print("========================================")
    print("        SIGNBRIDGE STARTING")
    print("========================================")
    print("MediaPipe version:", mp.__version__)
    print("Using MediaPipe Tasks API")
    print("Open: http://127.0.0.1:5000")
    print("========================================")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )
