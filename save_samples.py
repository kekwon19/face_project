from sklearn.datasets import fetch_lfw_people
from PIL import Image
import numpy as np
import os

lfw = fetch_lfw_people(min_faces_per_person=30, resize=1.0, color=True)
names = lfw.target_names

# 원하는 이름을 직접 지정 (기존 2명 + 여성 2명 추가)
target_people = ['Alejandro Toledo', 'Alvaro Uribe', 'Laura Bush', 'Serena Williams']

os.makedirs("people_photos", exist_ok=True)

for person in target_people:
    idx = list(names).index(person)
    person_folder = f"people_photos/{person.replace(' ', '_')}"
    os.makedirs(person_folder, exist_ok=True)

    count = 0
    for img_array, target_idx in zip(lfw.images, lfw.target):
        if target_idx == idx and count < 5:
            img = (img_array * 255).astype(np.uint8)
            Image.fromarray(img).save(f"{person_folder}/photo_{count}.jpg")
            count += 1
    print(f"{person}: {count}장 저장 완료 → {person_folder}")

print("완료!")