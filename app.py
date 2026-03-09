from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import cv2
import os
import uuid
import pickle
import requests
import pytesseract

from model_loader import load_model
from gradcam import generate_gradcam

# 🔹 SET YOUR TESSERACT PATH (Windows only)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 🔹 GOOGLE FACT CHECK API KEY
GOOGLE_API_KEY = "AIzaSyDrUzhbT6CjFG6GA7Hh2G5FNledMb4xW68"

app = Flask(__name__, static_folder="static")
CORS(app)

# 🔹 Load Image Model
model = load_model()

# 🔹 Load Fake News NLP Model
news_model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------- HOME ----------------
@app.route("/")
def home():
    return "Finderon Backend Running 🚀"

# ---------------- SERVE MARKED IMAGE ----------------
@app.route("/outputs/<filename>")
def get_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

# ---------------- FACT CHECK FUNCTION ----------------
def verify_fact(claim_text):
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    params = {
        "query": claim_text,
        "key": GOOGLE_API_KEY,
        "languageCode": "en"
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "claims" not in data:
        return None

    claim = data["claims"][0]
    review = claim["claimReview"][0]

    return {
        "claim": claim["text"],
        "publisher": review["publisher"]["name"],
        "rating": review["textualRating"],
        "review_url": review["url"]
    }

# ---------------- NEWS PREDICTION FUNCTION ----------------
def predict_news(text):
    text_vector = vectorizer.transform([text])
    prediction = news_model.predict(text_vector)[0]
    probability = news_model.predict_proba(text_vector)[0]

    confidence = max(probability) * 100

    if prediction == 1:
        result = "Real"
    else:
        result = "Fake"

    return {
        "result": result,
        "confidence": round(confidence, 2)
    }

# ---------------- MAIN ANALYZE ROUTE ----------------
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        filename = str(uuid.uuid4()) + ".jpg"

        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)

        original = cv2.imread(input_path)
        image = Image.open(input_path).convert("RGB")

        tensor = transform(image).unsqueeze(0)

        # 🔹 IMAGE AI PREDICTION
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probs, 1)

        image_result = "Fake" if pred.item() == 0 else "Real"
        image_confidence = round(confidence.item() * 100, 2)

        # 🔹 GRADCAM
        marked_image = generate_gradcam(model, tensor, original)
        output_name = "marked_" + filename
        output_path = os.path.join(OUTPUT_FOLDER, output_name)
        cv2.imwrite(output_path, marked_image)

        # 🔹 OCR TEXT EXTRACTION
        extracted_text = pytesseract.image_to_string(image)

        news_ai_result = None
        fact_result = None

        if extracted_text.strip() != "":
            news_ai_result = predict_news(extracted_text)
            fact_result = verify_fact(extracted_text)

        return jsonify({
            "image_result": image_result,
            "image_confidence": image_confidence,
            "extracted_text": extracted_text,
            "news_ai_prediction": news_ai_result,
            "fact_check": fact_result if fact_result else "No fact check found",
            "marked_image": f"/outputs/{output_name}"
        })

    except Exception as e:
        print("SERVER ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)