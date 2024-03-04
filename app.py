from flask import Flask, render_template, request, jsonify
from torchvision import models, transforms
from PIL import Image
import torch
import ast

app = Flask(__name__)

# Load pre-trained ResNet101 model
model = models.resnet101(pretrained=True)
model.eval()
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
        class_id = predicted.item()
        label = class_mappings.get(class_id, 'Unknown')
        return label

def load_class_mappings(file_path):
    with open(file_path, 'r') as file:
        class_mappings_str = file.read()
        class_mappings = ast.literal_eval(class_mappings_str)
    return class_mappings

class_mappings = load_class_mappings('class_labels.txt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        label = classify_image(file)
        return jsonify({'label': label})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
