import requests
import os
import uuid

SAVE_PATH = "../dataset/train/real"
os.makedirs(SAVE_PATH, exist_ok=True)

for i in range(200):
    url = "https://source.unsplash.com/random/800x600"
    response = requests.get(url)

    filename = os.path.join(SAVE_PATH, f"{uuid.uuid4()}.jpg")
    with open(filename, "wb") as f:
        f.write(response.content)

    print(f"Downloaded {i+1}")