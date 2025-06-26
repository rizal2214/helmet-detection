# Helmet Detection Flask Backend

Backend Flask untuk model binary image classification yang mendeteksi penggunaan helmet/safety helmet menggunakan EfficientNetB0.

## Fitur

-   ✅ Upload gambar untuk deteksi
-   ✅ Live camera capture dan analisis real-time
-   ✅ Web interface yang interaktif dan responsif
-   ✅ API endpoints untuk integrasi
-   ✅ Preprocessing gambar otomatis
-   ✅ Confidence score untuk setiap prediksi

## Struktur Project

helmet-detection-flask/
├── app.py
├── model/
│ ├── base_model.keras
│ ├── best_model.keras
│ ├── callback_augmentation_model.keras
│ └── helmet_detection_model.keras
├── static/
├── templates/
│ └── index.html
├── uploads/
├── .gitignore
├── .python-version
├── Procfile
├── README.md
└── requirements.txt

## Instalasi

### 1. Clone atau Download Project

```bash
# Buat folder project
mkdir helmet-detection
cd helmet-detection
```

### 2. Copy File Model

Pastikan file model `best_model.h5` Anda berada di root directory project.

### 3. Install Dependencies

```bash
# Install dependencies
pip install -r requirements.txt
```

Atau install manual:

```bash
pip install Flask==2.3.3 tensorflow==2.13.0 Pillow==10.0.1 numpy==1.24.3 opencv-python==4.8.1.78 Werkzeug==2.3.7
```

## Menjalankan Aplikasi

```bash
python app.py
```

Aplikasi akan berjalan di `http://localhost:5000`

## Cara Penggunaan

### 1. Upload Image

1. Buka browser dan akses `http://localhost:5000`
2. Pilih tab "Upload Image"
3. Drag & drop gambar atau klik "Choose Image"
4. Sistem akan otomatis menganalisis dan menampilkan hasil

### 2. Live Camera

1. Pilih tab "Live Camera"
2. Klik "Start Camera" (browser akan meminta permission)
3. Klik "Capture & Analyze" untuk mengambil foto dan menganalisis
4. Klik "Stop Camera" untuk menghentikan kamera
