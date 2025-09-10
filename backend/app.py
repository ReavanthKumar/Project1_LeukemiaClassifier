import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, request, jsonify
from PIL import Image
from flask_cors import CORS

# Azure Blob Storage
from azure.storage.blob import BlobClient

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Ensure static and model folders exist
if not os.path.exists("static"):
    os.makedirs("static")
if not os.path.exists("models"):
    os.makedirs("models")

# --------------------------
# 🔹 Azure Blob SAS URLs
# --------------------------
RESNET34_URL = "https://leukemiamodels.blob.core.windows.net/models/best_model_inceptionv3.pth?sp=r&st=2025-09-10T05:39:15Z&se=2026-09-05T13:54:15Z&spr=https&sv=2024-11-04&sr=b&sig=%2FsUNQzQMj%2FHEodWE7QENPxbAt5KPVi6j9a3%2FUhk0m8w%3D"
INCEPTION_URL = "https://leukemiamodels.blob.core.windows.net/models/best_model_resnet34.pth?sp=r&st=2025-09-10T05:40:18Z&se=2026-12-18T13:55:18Z&spr=https&sv=2024-11-04&sr=b&sig=0xD3lbb3PvN94yZrVTb60oPTKUGvB830Ki6A2iN79Zw%3D"

def download_model(url, local_path):
    """Download model from Azure Blob if not exists locally"""
    if not os.path.exists(local_path):
        print(f"⬇️ Downloading: {local_path}")
        blob_client = BlobClient.from_blob_url(url)
        with open(local_path, "wb") as f:
            f.write(blob_client.download_blob().readall())
    else:
        print(f"✅ Found cached model: {local_path}")

# Download models on startup
download_model(RESNET34_URL, "models/best_model_resnet34.pth")
download_model(INCEPTION_URL, "models/best_model_inceptionv3.pth")

# --------------------------
# 🔹 Config
# --------------------------
class_names = ["Benign", "Early", "Pre", "Pro"]
num_classes = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# 🔹 Load ResNet34
# --------------------------
resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
resnet34.fc = nn.Linear(resnet34.fc.in_features, num_classes)
resnet34.load_state_dict(torch.load("models/best_model_resnet34.pth", map_location=device))
resnet34.eval().to(device)

# --------------------------
# 🔹 Load Inception-v3
# --------------------------
inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
inception.fc = nn.Linear(inception.fc.in_features, num_classes)

if inception.AuxLogits is not None:
    inception.AuxLogits.fc = nn.Linear(inception.AuxLogits.fc.in_features, num_classes)

state_dict = torch.load("models/best_model_inceptionv3.pth", map_location=device)
inception.load_state_dict(state_dict)
inception = inception.to(device)
inception.eval()

# --------------------------
# 🔹 Transforms
# --------------------------
transform_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_incep = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --------------------------
# 🔹 Predict Endpoint
# --------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files or "model" not in request.form:
        return jsonify({"error": "Missing file or model"}), 400

    file = request.files["file"]
    model_choice = request.form["model"]

    file_path = os.path.join("static", file.filename)
    file.save(file_path)

    img = Image.open(file_path).convert("RGB")

    if model_choice == "resnet34":
        img_tensor = transform_resnet(img).unsqueeze(0).to(device)
        model = resnet34
    elif model_choice == "inceptionv3":
        img_tensor = transform_incep(img).unsqueeze(0).to(device)
        model = inception
    else:
        return jsonify({"error": "Invalid model choice"}), 400

    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        pred_class = class_names[preds.item()]

    return jsonify({"prediction": pred_class, "filename": file.filename})

if __name__ == "__main__":
    app.run(debug=True)

from flask import send_from_directory

# Serve React frontend
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    if path != "" and os.path.exists(os.path.join("frontend/build", path)):
        return send_from_directory("frontend/build", path)
    else:
        return send_from_directory("frontend/build", "index.html")
