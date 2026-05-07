import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from flask import Flask, render_template, Response, request, jsonify
from PIL import Image, ImageOps
import io
import base64
import matplotlib.pyplot as plt

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
    """
    path = os.path.join(MODELS_DIR, f"{model_name}.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

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
        # Fallback
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

def scan_models():
    if not os.path.exists(MODELS_DIR):
        return []
    files = [f[:-4] for f in os.listdir(MODELS_DIR) if f.endswith('.pth')]
    return sorted(files)

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
    if model_name is None:
        model_name = default_model
    if model_name not in loaded_models:
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
            img_bytes = file.read()
            if len(img_bytes) == 0:
                results.append({'filename': file.filename, 'error': 'File is empty'})
                continue
            img = Image.open(io.BytesIO(img_bytes))
            img = ImageOps.exif_transpose(img)
            img = img.convert('RGB')
            pred_class, conf, probs = predict_image(img, model_name)
            results.append({
                'filename': file.filename,
                'class': pred_class,
                'confidence': conf,
                'probabilities': probs
            })
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            results.append({
                'filename': file.filename,
                'error': str(e)
            })

    return jsonify(results)

# ================= SALIENCY MAP + ROI =================
def get_roi_bbox(saliency_map, threshold=0.3):
    blurred = cv2.GaussianBlur(saliency_map, (5, 5), 0)
    _, binary = cv2.threshold((blurred * 255).astype(np.uint8), int(threshold * 255), 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w < 0.05 * saliency_map.shape[1] or h < 0.05 * saliency_map.shape[0]:
        return None
    return (x, y, w, h)

def generate_saliency_map(pil_img, model, threshold=0.3, colormap_name='jet', roi_color=(0, 0, 255)):
    # Preprocess gambar seperti biasa
    img_tensor = preprocess_image(pil_img).to(DEVICE)
    img_tensor.requires_grad_()

    # Forward + backward untuk mendapatkan gradien
    outputs = model(img_tensor)
    pred_idx = outputs.argmax(dim=1).item()
    score = outputs[0, pred_idx]
    score.backward()

    # Ambil gradien, ambil maksimum absolut per channel
    grad = img_tensor.grad.data.abs()
    saliency, _ = grad.max(dim=1)
    saliency = saliency.cpu().numpy()[0]

    # Normalisasi ke 0-1
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    # Konversi gambar asli ke numpy (0-255)
    img_np = np.array(pil_img.resize(IMG_SIZE))

    # Buat heatmap berwarna sesuai colormap_name
    cmap = plt.get_cmap(colormap_name)
    heatmap = cmap(saliency)[..., :3] * 255
    heatmap = heatmap.astype(np.uint8)

    # Overlay dengan bobot 0.5
    overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

    # ROI bounding box dengan threshold yang bisa diatur
    bbox = get_roi_bbox(saliency, threshold)
    roi_img = img_np.copy()
    roi_img_bgr = cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR)  # konversi ke BGR untuk OpenCV
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(roi_img_bgr, (x, y), (x+w, y+h), roi_color, 2)
        cv2.putText(roi_img_bgr, "Focus", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 2)
    roi_img = cv2.cvtColor(roi_img_bgr, cv2.COLOR_BGR2RGB)  # kembalikan ke RGB
    explanation = ""
    if bbox:
        explanation = f"Model berfokus pada area {w}x{h} piksel (threshold={threshold:.2f})."
    else:
        explanation = "Model tidak memiliki fokus yang kuat (coba turunkan threshold)."

    return overlay, roi_img, pred_idx, torch.softmax(outputs, dim=1).cpu().detach().numpy()[0], explanation

@app.route('/saliency', methods=['POST'])
def saliency():
    data = request.get_json()
    model_name = data.get('model', default_model)
    img_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    # Parameter baru
    threshold = float(data.get('threshold', 0.3))
    colormap_name = data.get('colormap', 'jet')
    roi_color_str = data.get('roi_color', 'red')

    # Konversi nama warna ke BGR
    color_map = {
        'red':    (0, 0, 255),   # BGR: Blue=0, Green=0, Red=255
        'green':  (0, 255, 0),   # BGR: Blue=0, Green=255, Red=0
        'blue':   (255, 0, 0),   # BGR: Blue=255, Green=0, Red=0
        'yellow': (0, 255, 255), # BGR: Blue=0, Green=255, Red=255
        'cyan':   (255, 255, 0), # BGR: Blue=255, Green=255, Red=0
        'orange': (0, 165, 255), # BGR: Blue=0, Green=165, Red=255
        'white':  (255, 255, 255),
    }
    roi_color = color_map.get(roi_color_str, (0, 0, 255))  # default merah

    if model_name not in loaded_models:
        model_name = default_model
    model = loaded_models[model_name]

    try:
        overlay, roi_img, pred_idx, probs, explanation = generate_saliency_map(
            img, model, threshold=threshold, colormap_name=colormap_name, roi_color=roi_color
        )
        _, buffer_overlay = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        heatmap_b64 = base64.b64encode(buffer_overlay).decode('utf-8')
        _, buffer_roi = cv2.imencode('.jpg', cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR))
        roi_b64 = base64.b64encode(buffer_roi).decode('utf-8')

        return jsonify({
            'heatmap': heatmap_b64,
            'roi': roi_b64,
            'explanation': explanation,
            'class': CLASS_NAMES[pred_idx],
            'confidence': float(probs[pred_idx]),
            'probabilities': probs.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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