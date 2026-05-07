# 🎨 Undertone Scanner — Real‑time Skin Undertone Classification from Veins

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Flask](https://img.shields.io/badge/flask-3.0-green.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.2-red.svg)
![TailwindCSS](https://img.shields.io/badge/tailwindcss-3.4-38bdf8.svg)

**Undertone Scanner** adalah aplikasi web real‑time yang mengklasifikasikan *undertone* kulit (Cool, Neutral, Warm) dari gambar pergelangan tangan menggunakan **deep learning** dan **saliency map** untuk interpretasi visual.

Dibangun dengan **Flask**, **PyTorch**, dan **Tailwind CSS**, aplikasi ini mendukung perbandingan multi‑model, validasi label, dan analisis fokus model (ROI bounding box + heatmap).

---

## 🖥️ Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| 📤 **Upload Gambar** | Prediksi dua model sekaligus, bandingkan hasil secara berdampingan. |
| 📸 **Live Camera** | Tangkap gambar dari webcam, langsung dapatkan prediksi. |
| 📚 **Bulk Upload** | Proses puluhan gambar sekaligus, lihat hasil dalam grid beserta statistik akurasi per model. |
| 🧠 **Multi‑Model** | Muat otomatis semua model `.pth` dalam folder `models/`, pilih dua model untuk dibandingkan. |
| 🔬 **Advanced Result** *(Saliency Map)* | Lihat heatmap fokus model, bounding box ROI, dan penjelasan; kontrol interaktif untuk threshold, colormap, dan warna kotak. |
| ✅ **Expected Label** | Jika label sebenarnya diketahui, hasil prediksi akan di‑highlight **hijau** (benar) atau **merah** (salah). |
| ⚖️ **Statistik Akurasi Bulk** | Menghitung persentase benar/salah Model A vs Model B saat *expected label* diisi. |

---

## 🧠 Model yang Tersedia

Aplikasi secara otomatis memuat **13 model** dari folder `models/`.  
Model dilatih dengan dua mode preprocessing pada **5 versi dataset (v1–v5)**.

| Nama File | Arsitektur | Dataset | Preprocessing |
|-----------|------------|---------|---------------|
| `Mobilenetv2 datav1 CLAHE.pth` | MobileNetV2 | v1 | Segmentasi + CLAHE |
| `Mobilenetv2 datav1 No CLAHE.pth` | MobileNetV2 | v1 | Hanya resize |
| `Mobilenetv2 datav2 CLAHE.pth` | MobileNetV2 | v2 | Segmentasi + CLAHE |
| `Mobilenetv2 datav2 No CLAHE.pth` ⭐ | MobileNetV2 | v2 | Hanya resize |
| `Mobilenetv2 datav3 CLAHE.pth` | MobileNetV2 | v3 | Segmentasi + CLAHE |
| `Mobilenetv2 datav3 No CLAHE.pth` | MobileNetV2 | v3 | Hanya resize |
| `Mobilenetv2 datav4 CLAHE.pth` | MobileNetV2 | v4 | Segmentasi + CLAHE |
| `Mobilenetv2 datav4 No CLAHE.pth` | MobileNetV2 | v4 | Hanya resize |
| `Mobilenetv2 datav5 CLAHE.pth` | MobileNetV2 | v5 | Segmentasi + CLAHE |
| `Mobilenetv2 datav5 No CLAHE.pth` | MobileNetV2 | v5 | Hanya resize |
| `Resnet18 datav2 CLAHE.pth` | ResNet18 | v2 | Segmentasi + CLAHE |
| `Resnet18 datav2 No CLAHE.pth` | ResNet18 | v2 | Hanya resize |
| `Resnet50 datav2 CLAHE.pth` | ResNet50 | v2 | Segmentasi + CLAHE |

⭐ = Model dengan akurasi tertinggi (MobileNetV2 + v2 + tanpa CLAHE)

---

## 🏗️ Struktur Proyek

```
web_app/
├── app.py                  # Flask server + PyTorch inference
├── models/                 # File bobot model (.pth)
│   ├── Mobilenetv2 datav1 CLAHE.pth
│   ├── Mobilenetv2 datav2 No CLAHE.pth
│   ├── Resnet18 datav2 CLAHE.pth
│   └── ...
├── templates/
│   └── index.html          # UI utama (Tailwind CSS + JavaScript)
└── README.md
```

> **Catatan:** Script training (`pipeline_preprocess_dl.py`) dan dataset **tidak** disertakan dalam repositori ini.  
> Repositori ini hanya berisi aplikasi web siap‑pakai.

---

## 🚀 Cara Menjalankan

### 1. Clone Repositori
```bash
git clone https://github.com/Luthfanajwah/Undertone_Scanner.git
cd Undertone_Scanner/web_app
```

### 2. Install Dependensi
```bash
pip install flask torch torchvision pillow opencv-python matplotlib scikit-learn
```

### 3. Jalankan Server
```bash
python app.py
```

Buka browser ke **http://127.0.0.1:5050** (atau alamat IP lokal yang muncul di terminal).

Aplikasi akan otomatis memuat semua model `.pth` dari folder `models/`.

---

## 📊 Performa Model (Semua Eksperimen)

Berikut hasil evaluasi lengkap pada **test set** masing‑masing versi dataset.

| Model | Dataset | Preprocessing | Test Acc | Cool F1 | Neutral F1 | Warm F1 | Waktu Training |
|-------|---------|---------------|----------|---------|------------|---------|----------------|
| MobileNetV2 | v1 | `seg_clahe` | 71.83% | 0.71 | 0.64 | 0.79 | 52 menit |
| **MobileNetV2** | **v1** | **`none`** | **73.24%** | **0.79** | 0.63 | **0.79** | 44 menit |
| MobileNetV2 | v2 | `seg_clahe` | 74.65% | 0.76 | 0.68 | 0.81 | 32 menit |
| **MobileNetV2** ⭐ | **v2** | **`none`** | **76.06%** | 0.77 | 0.65 | **0.84** | 27 menit |
| ResNet18 | v2 | `seg_clahe` | 74.65% | **0.79** | 0.65 | 0.81 | 36 menit |
| ResNet18 | v2 | `none` | 69.01% | 0.67 | 0.59 | 0.78 | 39 menit |
| ResNet50 | v2 | `seg_clahe` | 64.79% | 0.58 | 0.56 | 0.75 | 119 menit |
| MobileNetV2 | v3 | `seg_clahe` | 66.89% | 0.73 | 0.59 | 0.71 | 60 menit |
| MobileNetV2 | v3 | `none` | 70.86% | 0.72 | 0.69 | 0.73 | 68 menit |
| MobileNetV2 | v4 | `seg_clahe` | 69.54% | 0.75 | 0.68 | 0.69 | 49 menit |
| MobileNetV2 | v4 | `none` | 69.54% | 0.72 | 0.70 | 0.68 | 29 menit |
| MobileNetV2 | v5 | `seg_clahe` | 61.25% | 0.54 | 0.67 | 0.55 | 27 menit |
| MobileNetV2 | v5 | `none` | 57.50% | 0.57 | 0.63 | 0.50 | 25 menit |

⭐ **Model unggulan:** `Mobilenetv2 datav2 No CLAHE.pth`  
- Akurasi tertinggi (76.06%) dengan waktu training tercepat (27 menit)  
- F1 Warm tertinggi (0.84)  
- Arsitektur paling ringan, cocok untuk deployment CPU

### 🔍 Temuan Utama
- **MobileNetV2 + tanpa CLAHE** secara konsisten mengungguli mode `seg_clahe` di dataset yang sama.
- **ResNet18** hanya kompetitif jika preprocessing `seg_clahe` digunakan, namun kalah efisien.
- **ResNet50** tidak direkomendasikan (akurasi rendah, waktu training >2 jam).
- **Dataset v2** memberikan keseimbangan terbaik antara ukuran, kualitas, dan akurasi.
- **Dataset v5** terlalu kecil (833 gambar) sehingga akurasi jeblok.

---

## 🔧 API Endpoints

| Method | Route | Deskripsi |
|--------|-------|-----------|
| `GET` | `/` | Halaman utama |
| `GET` | `/get_models` | Mendapatkan daftar model yang tersedia |
| `POST` | `/predict_upload` | Prediksi gambar yang di‑upload (multipart form) |
| `POST` | `/predict_frame` | Prediksi frame kamera (base64 JSON) |
| `POST` | `/predict_bulk` | Prediksi banyak gambar sekaligus |
| `POST` | `/saliency` | Menghasilkan saliency map + ROI |
| `GET` | `/video_feed` | Streaming webcam dengan overlay prediksi |

---

## 🛠️ Teknologi yang Digunakan

- **Backend:** Flask, PyTorch, Torchvision, OpenCV, PIL
- **Frontend:** Tailwind CSS (CDN), Material Symbols, Vanilla JavaScript
- **Model:** MobileNetV2, ResNet18, ResNet50 (Transfer Learning dari ImageNet)
- **Visualisasi:** Matplotlib (colormap), Saliency Map, t‑SNE (opsional)

---

**🎯 Undertone Scanner — combining computer vision and explainability for real‑time skin analysis.**
