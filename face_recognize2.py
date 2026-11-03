import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from sklearn.neighbors import KNeighborsClassifier

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

def get_embedding(path):
    img = cv2.imread(path)
    faces = app.get(img)
    if len(faces) == 0:
        print(f"  ⚠ 얼굴을 못 찾음: {path}")
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

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)
print("분류기 학습 완료, 웹캠 시작!")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        box = face.bbox.astype(int)
        embedding = face.embedding

        pred = model.predict([embedding])[0]
        prob = max(model.predict_proba([embedding])[0])

        if prob < 0.5:
            label = "Unknown"
            color = (0, 0, 255)
        else:
            label = f"{pred} ({prob*100:.0f}%)"
            color = (0, 255, 0)

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(frame, label, (box[0], box[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()