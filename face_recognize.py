import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_lfw_people

# 1. 얼굴 분석기 준비
print("1. 얼굴 분석기 준비 중...")
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

def get_embedding(path):
    img = cv2.imread(path)
    faces = app.get(img)
    if len(faces) == 0:
        return None
    return faces[0].embedding

# 2. 학습 데이터 준비 (아까와 동일)
print("2. 학습 데이터 준비 중...")
my_embeddings = []
for filename in os.listdir("my_photos"):
    emb = get_embedding(os.path.join("my_photos", filename))
    if emb is not None:
        my_embeddings.append(emb)

lfw = fetch_lfw_people(min_faces_per_person=5, resize=1.0, color=True)
other_embeddings = []
for img_array in lfw.images[:100]:
    img = (img_array * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    faces = app.get(img)
    if len(faces) > 0:
        other_embeddings.append(faces[0].embedding)

X = np.array(my_embeddings + other_embeddings)
y = np.array([1] * len(my_embeddings) + [0] * len(other_embeddings))

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)
print("3. 분류기 학습 완료, 웹캠 시작!")

# 3. 웹캠 실시간 인식
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
        prob = model.predict_proba([embedding])[0]

        if pred == 1:
            label = f"Me ({prob[1]*100:.0f}%)"
            color = (0, 255, 0)  # 초록
        else:
            label = f"Unknown ({prob[0]*100:.0f}%)"
            color = (0, 0, 255)  # 빨강

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(frame, label, (box[0], box[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()