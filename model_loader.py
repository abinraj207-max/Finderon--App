import random

def predict_image(image):
    # Temporary lightweight demo logic
    result = random.choice(["Real", "Fake"])
    confidence = round(random.uniform(70, 99), 2)

    return result, confidence