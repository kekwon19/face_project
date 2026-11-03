import cv2
from insightface.app import FaceAnalysis

# 얼굴 분석기 준비 (처음 실행 시 모델 자동 다운로드, 시간 좀 걸릴 수 있음)
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# 웹캠 열기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        box = face.bbox.astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

