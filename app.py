from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import cv2
import os
import uuid

from model_loader import load_model
from gradcam import generate_gradcam

app = Flask(__name__)
CORS(app)

model = load_model()

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.route("/")
def home():
    return "Finderon Backend Running 🚀"

@app.route("/outputs/<filename>")
def get_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

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

        # Prediction
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probs, 1)

        result = "Fake" if pred.item() == 0 else "Real"
        confidence_percent = round(confidence.item() * 100, 2)

        explanation = (
            "Model analyzed global pixel patterns and texture inconsistencies. "
            "Highlighted regions show strongest activation."
        )

        # GradCAM
        marked_image = generate_gradcam(model, tensor, original)

        output_name = "marked_" + filename
        output_path = os.path.join(OUTPUT_FOLDER, output_name)

        cv2.imwrite(output_path, marked_image)

        return jsonify({
            "result": result,
            "confidence": confidence_percent,
            "explanation": explanation,
            "marked_image": f"/outputs/{output_name}"
        })

    except Exception as e:
        print("SERVER ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)