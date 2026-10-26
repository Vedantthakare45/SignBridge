"""
Train an A-Z classifier from data/alphabet_landmarks.csv.

Run:
    python train_model.py

Creates:
    model/alphabet_model.joblib

The browser prototype currently uses a small demo classifier so it can run
without a binary model. This trained model can be integrated into a Python
OpenCV inference service for the production version.
"""
import os, joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DATA="data/alphabet_landmarks.csv"
OUT="model/alphabet_model.joblib"

if not os.path.exists(DATA):
    raise SystemExit("Dataset not found. Run collect_data.py first.")

df=pd.read_csv(DATA)
X=df.drop(columns=["label"])
y=df["label"]

if y.nunique()<2:
    raise SystemExit("Collect at least two different letters before training.")

Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model=RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1)
model.fit(Xtr,ytr)
pred=model.predict(Xte)

print("Accuracy:",accuracy_score(yte,pred))
print(classification_report(yte,pred))
os.makedirs("model",exist_ok=True)
joblib.dump(model,OUT)
print("Saved:",OUT)
