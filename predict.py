"""
Standalone webcam prediction using a trained A-Z model.

Run after training:
    python predict.py
"""
import cv2, joblib, mediapipe as mp
import numpy as np

MODEL="model/alphabet_model.joblib"
model=joblib.load(MODEL)
hands_mod=mp.solutions.hands
draw=mp.solutions.drawing_utils

cap=cv2.VideoCapture(0)

with hands_mod.Hands(max_num_hands=1,min_detection_confidence=.6,min_tracking_confidence=.6) as hands:
    while True:
        ok,frame=cap.read()
        if not ok:break
        res=hands.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        label="—"

        if res.multi_hand_landmarks:
            lm=res.multi_hand_landmarks[0]
            pts=np.array([[p.x,p.y,p.z] for p in lm.landmark],dtype=np.float32)
            pts-=pts[0]
            pts/=np.linalg.norm(pts[9]) or 1
            pred=model.predict([pts.flatten()])[0]
            label=str(pred)
            draw.draw_landmarks(frame,lm,hands_mod.HAND_CONNECTIONS)

        cv2.putText(frame,f"Letter: {label}",(20,50),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),3)
        cv2.imshow("SignBridge A-Z Model",frame)
        if cv2.waitKey(1)&0xFF==ord("q"):break

cap.release();cv2.destroyAllWindows()
