import os 
import torch 
import torch.nn as nn 
import torchvision.transforms as transforms 
from torchvision import models 
from flask import Flask, request, jsonify 
from PIL import Image 
from flask import Flask, request, jsonify, render_template 
from flask_cors import CORS 

app = Flask(__name__) 
CORS(app) 

# Enable CORS for all routes 
#  Ensure static folder exists if not 
os.path.exists("static"): os.makedirs("static") 
#Class names 
class_names = ["Benign", "Early", "Pre", "Pro"] 
num_classes = 4 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Load ResNet34 
resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1) 
resnet34.fc = nn.Linear(resnet34.fc.in_features, num_classes) 
resnet34.load_state_dict(torch.load("models/best_model_resnet34.pth", map_location=device)) 
resnet34.eval().to(device) 
# Load Inception-v3 
import torch 
import torch.nn as nn 
from torchvision import models 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
num_classes = 4 

# Benign, Early, Pre, Pro (adjust as per your dataset) 
# # Load inception v3 with aux_logits=True (default) 

inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True) 
# Replace main classifier 
inception.fc = nn.Linear(inception.fc.in_features, num_classes) 
# Replace auxiliary classifier (since you did this during training) 
if inception.AuxLogits is not None: 
    inception.AuxLogits.fc = nn.Linear(inception.AuxLogits.fc.in_features, num_classes) 
# Load trained weights 
state_dict = torch.load("models/best_model_inceptionv3.pth", map_location=device) 
inception.load_state_dict(state_dict) 
inception = inception.to(device) 
inception.eval() 
# Transforms 
transform_resnet = transforms.Compose([ transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ]) 
transform_incep = transforms.Compose([ transforms.Resize((299, 299)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ]) 

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
