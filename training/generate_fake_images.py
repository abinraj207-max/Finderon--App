import cv2
import os
import random
import uuid

REAL_PATH = "../dataset/train/real"
FAKE_PATH = "../dataset/train/fake"

os.makedirs(FAKE_PATH, exist_ok=True)

for img_name in os.listdir(REAL_PATH):
    img_path = os.path.join(REAL_PATH, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    h, w, _ = img.shape

    x1 = random.randint(0, w//2)
    y1 = random.randint(0, h//2)
    x2 = random.randint(w//2, w)
    y2 = random.randint(h//2, h)

    img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (51,51), 0)

    cv2.putText(img, "BREAKING FAKE NEWS", (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,0,255), 3)

    fake_name = os.path.join(FAKE_PATH, f"{uuid.uuid4()}.jpg")
    cv2.imwrite(fake_name, img)

print("Fake images generated.")