import cv2
import numpy as np
import os
import joblib
from insightface.app import FaceAnalysis
from sklearn.neighbors import KNeighborsClassifier

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

def get_embedding(path):
    img = cv2.imread(path)
    faces = app.get(img)
    if len(faces) == 0:
        return None
    return faces[0].embedding

base_folder = "people_photos"
X = []
y = []

for person_name in os.listdir(base_folder):
    person_folder = os.path.join(base_folder, person_name)
    if not os.path.isdir(person_folder):
        continue
    for filename in os.listdir(person_folder):
        emb = get_embedding(os.path.join(person_folder, filename))
        if emb is not None:
            X.append(emb)
            y.append(person_name)

print(f"학습 데이터 준비 완료: {len(X)}장, 등록된 사람: {set(y)}")

# 학습
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)
print("분류기 학습 완료")

# 학습된 모델을 파일로 저장
joblib.dump(model, "face_model.pkl")
print("모델을 face_model.pkl 파일로 저장 완료!")