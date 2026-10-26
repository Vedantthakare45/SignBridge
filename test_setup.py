import sys
import mediapipe as mp

print("Python:", sys.version)
print("MediaPipe:", mp.__version__)
print("Has new Tasks API:", hasattr(mp, "tasks"))
print("Has old solutions API:", hasattr(mp, "solutions"))

if hasattr(mp, "tasks"):
    print("OK: current MediaPipe Tasks API is available.")
else:
    print("ERROR: MediaPipe Tasks API is missing.")
