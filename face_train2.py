import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. 얼굴 임베딩 추출기 준비
print("1. 얼굴 분석기 준비 중...")
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# 2. 사진 한 장에서 얼굴 임베딩(벡터) 뽑아내는 함수
def get_embedding(path):
    img = cv2.imread(path)
    faces = app.get(img)
    if len(faces) == 0:
        print(f"  ⚠ 얼굴을 못 찾음: {path}")
        return None
    return faces[0].embedding  # 512차원 벡터

# 3. 내 사진들에서 임베딩 뽑기
print("2. 내 사진 임베딩 추출 중...")
my_embeddings = []
for filename in os.listdir("my_photos"):
    path = os.path.join("my_photos", filename)
    emb = get_embedding(path)
    if emb is not None:
        my_embeddings.append(emb)

print(f"   내 사진 {len(my_embeddings)}장에서 임베딩 추출 완료")

# 4. "남" 데이터 필요 - Olivetti 대신, 얼굴 여러 장이 있는 공개 데이터셋 사용
from sklearn.datasets import fetch_lfw_people
print("3. 비교용 타인 얼굴 데이터 불러오는 중...")
lfw = fetch_lfw_people(min_faces_per_person=5, resize=1.0, color=True)

other_embeddings = []
for img_array in lfw.images[:100]:  # 속도를 위해 100장만 사용
    img = (img_array * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    faces = app.get(img)
    if len(faces) > 0:
        other_embeddings.append(faces[0].embedding)

print(f"   타인 임베딩 {len(other_embeddings)}개 추출 완료")

# 5. 데이터 합치고 학습
X = np.array(my_embeddings + other_embeddings)
y = np.array([1] * len(my_embeddings) + [0] * len(other_embeddings))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
print("4. 분류기 학습 완료")

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"5. 테스트 정확도: {accuracy * 100:.1f}%")

# 6. 새 사진으로 검증
new_emb = get_embedding("test_me.jpg")
if new_emb is not None:
    pred = model.predict([new_emb])[0]
    prob = model.predict_proba([new_emb])[0]
    if pred == 1:
        print(f"6. 새 사진 예측: 본인입니다! (확신도 {prob[1]*100:.1f}%)")
    else:
        print(f"6. 새 사진 예측: 본인이 아닙니다. (확신도 {prob[0]*100:.1f}%)")