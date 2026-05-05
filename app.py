import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from flask import Flask, render_template, Response, request, jsonify
from PIL import Image
import io
import base64

# ================= KONFIGURASI =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
IMG_SIZE = (224, 224)
CLASS_NAMES = ['cool', 'neutral', 'warm']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LOAD ALL MODELS =================
loaded_models = {}

def load_single_model(model_name):
    """
    Load satu model berdasarkan nama file (tanpa .pth).
    Arsitektur otomatis dideteksi dari nama model (case-insensitive).
    - mengandung 'resnet18' -> ResNet18
    - mengandung 'resnet50' -> ResNet50
    - mengandung 'mobilenet' -> MobileNetV2
    Kalau tidak cocok, akan dicoba sebagai MobileNetV2 dulu, lalu ResNet18.
    """
    path = os.path.join(MODELS_DIR, f"{model_name}.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    # Normalisasi: lowercase, tanpa spasi
    key = model_name.replace(' ', '').lower()

    if 'resnet18' in key:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    elif 'resnet50' in key:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    elif 'mobilenet' in key:
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
    else:
        # Fallback: coba satu per satu
        try:
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            return model
        except RuntimeError:
            try:
                model = models.resnet18(weights=None)
                model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
                model.load_state_dict(torch.load(path, map_location=DEVICE))
                model.to(DEVICE)
                model.eval()
                return model
            except RuntimeError:
                raise RuntimeError(
                    f"Tidak bisa mengenali arsitektur model '{model_name}'. "
                    f"Pastikan nama file mengandung 'mobilenet', 'resnet18', atau 'resnet50'."
                )

    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# Pindai semua model
def scan_models():
    if not os.path.exists(MODELS_DIR):
        return []
    files = [f[:-4] for f in os.listdir(MODELS_DIR) if f.endswith('.pth')]
    return sorted(files)


# Jalankan saat startup
print("Scanning models...")
available_models = scan_models()
if not available_models:
    raise RuntimeError(f"No .pth files found in {MODELS_DIR}")

for name in available_models:
    print(f"Loading {name}...")
    loaded_models[name] = load_single_model(name)

default_model = available_models[0]
print(f"All models loaded. Default: {default_model}")

# ================= PREPROCESSING =================
preprocess = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(pil_img):
    return preprocess(pil_img).unsqueeze(0).to(DEVICE)

def predict_image(pil_img, model_name=None):
    """Prediksi menggunakan model tertentu (default jika None)."""
    if model_name is None:
        model_name = default_model
    if model_name not in loaded_models:
        # fallback
        model_name = default_model
    model = loaded_models[model_name]
    img_tensor = preprocess_image(pil_img)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    class_idx = np.argmax(probs)
    confidence = float(probs[class_idx])
    class_name = CLASS_NAMES[class_idx]
    return class_name, confidence, probs.tolist()

# ================= FLASK APP =================
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', class_names=CLASS_NAMES, models=available_models, default_model=default_model)

@app.route('/get_models')
def get_models():
    return jsonify(available_models)

@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    model_name = request.form.get('model', default_model)
    img = Image.open(file.stream).convert('RGB')
    pred_class, conf, all_probs = predict_image(img, model_name)
    return jsonify({
        'class': pred_class,
        'confidence': conf,
        'probabilities': all_probs,
        'model': model_name
    })

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    data = request.get_json()
    model_name = data.get('model', default_model)
    img_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    pred_class, conf, all_probs = predict_image(img, model_name)
    return jsonify({
        'class': pred_class,
        'confidence': conf,
        'probabilities': all_probs,
        'model': model_name
    })

@app.route('/predict_bulk', methods=['POST'])
def predict_bulk():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files')
    model_name = request.form.get('model', default_model)
    results = []

    for file in files:
        if file.filename == '':
            continue
        try:
            # Baca bytes agar stream tidak habis
            img_bytes = file.read()
            if len(img_bytes) == 0:
                results.append({'filename': file.filename, 'error': 'File is empty'})
                continue

            # Buka gambar dari bytes
            img = Image.open(io.BytesIO(img_bytes))

            # Transpose orientasi EXIF
            from PIL import ImageOps
            img = ImageOps.exif_transpose(img)

            # Konversi ke RGB
            img = img.convert('RGB')

            pred_class, conf, probs = predict_image(img, model_name)
            results.append({
                'filename': file.filename,
                'class': pred_class,
                'confidence': conf,
                'probabilities': probs
            })
        except Exception as e:
            # Log error untuk debugging
            print(f"Error processing {file.filename}: {e}")
            results.append({
                'filename': file.filename,
                'error': str(e)
            })

    return jsonify(results)

# ================= STREAMING (Opsional) =================
def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        img = cv2.resize(frame, IMG_SIZE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pred_class, conf, _ = predict_image(pil_img, default_model)
        cv2.putText(frame, f"{pred_class} ({conf:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)