import cv2
import joblib
from insightface.app import FaceAnalysis

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# 학습 코드 없이, 저장된 모델 파일만 불러오기
model = joblib.load("face_model.pkl")
print("저장된 모델 불러오기 완료!")

# 바로 웹캠 실행 (학습 과정 없음!)
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