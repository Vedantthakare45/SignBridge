from pathlib import Path
from urllib.request import urlopen

URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
OUT = Path(__file__).parent / "assets" / "hand_landmarker.task"

OUT.parent.mkdir(parents=True, exist_ok=True)

print("Downloading MediaPipe hand model...")
with urlopen(URL, timeout=60) as response:
    data = response.read()

OUT.write_bytes(data)
print("Saved:", OUT)
print("Size:", len(data), "bytes")
