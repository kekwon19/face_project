from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. 얼굴 데이터 불러오기 (400장, 40명, 각 10장씩)
print("1. 데이터 불러오는 중...")
data = fetch_olivetti_faces()
X = data.images.reshape(len(data.images), -1)  # 이미지를 1줄짜리 숫자 배열로 펼치기
y = data.target  # 각 사진이 몇 번 사람인지 라벨
print(f"2. 데이터 {len(X)}장, {len(set(y))}명 불러오기 완료")

# 2. 학습용/테스트용으로 나누기
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"3. 학습용 {len(X_train)}장, 테스트용 {len(X_test)}장으로 분리")

# 3. 분류기 직접 학습시키기 (여기가 핵심!)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
print("4. 분류기 학습 완료")

# 4. 테스트 데이터로 성능 확인
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"5. 정확도: {accuracy * 100:.1f}%")

import cv2
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os

# 1. Olivetti 데이터 불러오기 (64x64 흑백)
print("1. 기존 데이터 불러오는 중...")
data = fetch_olivetti_faces()
X_others = data.images  # (400, 64, 64), 값 범위 0~1

# 2. 내 사진 불러와서 Olivetti와 같은 형식(64x64 흑백, 0~1)으로 맞추기
def load_and_prepare(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 흑백으로 읽기
    img = cv2.resize(img, (64, 64))                # 크기 통일
    img = img / 255.0                              # 0~1 범위로 정규화
    return img

my_folder = "my_photos"
my_images = []
for filename in os.listdir(my_folder):
    path = os.path.join(my_folder, filename)
    img = load_and_prepare(path)
    my_images.append(img)

print(f"2. 내 사진 {len(my_images)}장 불러옴")

# 3. 데이터 증강: 좌우 반전 + 밝기 변형으로 개수 늘리기
augmented = []
for img in my_images:
    augmented.append(img)                          # 원본
    augmented.append(cv2.flip(img, 1))              # 좌우 반전
    augmented.append(np.clip(img * 1.2, 0, 1))      # 밝게
    augmented.append(np.clip(img * 0.8, 0, 1))      # 어둡게

my_images_final = np.array(augmented)
print(f"3. 증강 후 내 사진 {len(my_images_final)}장으로 늘림")

# 4. 전체 데이터 합치기 (나=1, 남=0)
X = np.concatenate([X_others.reshape(len(X_others), -1),
                     my_images_final.reshape(len(my_images_final), -1)])
y = np.array([0] * len(X_others) + [1] * len(my_images_final))

# 5. 학습/테스트 분리 후 학습
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
print("4. '나 vs 남' 분류기 학습 완료")

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"5. 정확도: {accuracy * 100:.1f}%")

# 6. 학습에 안 쓴 새 사진으로 진짜 테스트
def predict_new_photo(path, model):
    img = load_and_prepare(path)          # 아까 만든 함수 재사용
    img_flat = img.reshape(1, -1)          # 모델에 넣을 수 있는 형태로 펼치기
    prediction = model.predict(img_flat)[0]
    prob = model.predict_proba(img_flat)[0]
    return prediction, prob

result, prob = predict_new_photo("test_me.jpg", model)

if result == 1:
    print(f"6. 예측 결과: 본인입니다! (확신도: {prob[1]*100:.1f}%)")
else:
    print(f"6. 예측 결과: 본인이 아닙니다. (확신도: {prob[0]*100:.1f}%)")