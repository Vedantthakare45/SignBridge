"""
Collect A-Z hand landmark data for training.

Run:
    python collect_data.py

Controls:
    A-Z : choose label
    S   : save one sample
    Q   : quit

The output is data/alphabet_landmarks.csv

For a better model, collect many samples per letter with:
- different people
- different hand sizes
- different backgrounds
- different lighting
- small rotations/positions
"""
import csv, os, cv2, mediapipe as mp
import numpy as np

OUT="data/alphabet_landmarks.csv"
os.makedirs("data",exist_ok=True)

mp_hands=mp.solutions.hands
mp_draw=mp.solutions.drawing_utils

cap=cv2.VideoCapture(0)
current="A"

with mp_hands.Hands(static_image_mode=False,max_num_hands=1,min_detection_confidence=.6,min_tracking_confidence=.6) as hands:
    while True:
        ok,frame=cap.read()
        if not ok: break
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res=hands.process(rgb)
        cv2.putText(frame,f"Label: {current} | S=save Q=quit",(20,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,0),2)

        if res.multi_hand_landmarks:
            lm=res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame,lm,mp_hands.HAND_CONNECTIONS)
            pts=np.array([[p.x,p.y,p.z] for p in lm.landmark],dtype=np.float32)
            pts-=pts[0]
            scale=np.linalg.norm(pts[9]) or 1
            pts/=scale
            flat=pts.flatten().tolist()
        else: flat=None

        cv2.imshow("SignBridge Data Collection",frame)
        key=cv2.waitKey(1)&0xFF
        if key==ord('q'): break
        if 65<=key<=90: current=chr(key)
        if key==ord('s') and flat:
            new=not os.path.exists(OUT)
            with open(OUT,"a",newline="",encoding="utf-8") as f:
                w=csv.writer(f)
                if new:w.writerow(["label"]+[f"f{i}" for i in range(len(flat))])
                w.writerow([current]+flat)
            print("Saved",current)

cap.release();cv2.destroyAllWindows()
