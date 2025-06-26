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

```
helmet-detection-flask/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Web interface
├── uploads/              # Folder untuk temporary uploads
├── best_model.h5         # Model Anda (.h5 file)
├── requirements.txt      # Python dependencies
└── README.md            # Dokumentasi ini
```

## Instalasi

### 1. Clone atau Download Project

```bash
# Buat folder project
mkdir helmet-detection-flask
cd helmet-detection-flask
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

### 4. Buat Folder Templates

```bash
mkdir templates
```

### 5. Copy File HTML

Copy file `index.html` ke folder `templates/`.

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

## API Endpoints

### 1. Upload Image Prediction

```
POST /predict/upload
Content-Type: multipart/form-data

Body: file (image file)
```

Response:

```json
{
    "predicted_class": 1,
    "class_name": "with_helmet",
    "confidence": 87.45,
    "raw_prediction": 0.8745
}
```

### 2. Camera Image Prediction

```
POST /predict/camera
Content-Type: application/json

Body: {
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
}
```

Response:

```json
{
    "predicted_class": 0,
    "class_name": "without_helmet",
    "confidence": 92.31,
    "raw_prediction": 0.0769
}
```

### 3. Health Check

```
GET /health
```

Response:

```json
{
    "status": "healthy",
    "model_loaded": true,
    "message": "Helmet Detection API is running"
}
```

## Class Labels

-   **0**: `without_helmet` - Tidak menggunakan helmet
-   **1**: `with_helmet` - Menggunakan helmet

## Spesifikasi Model

-   **Architecture**: EfficientNetB0 (pre-trained)
-   **Input Size**: 224x224x3 (RGB)
-   **Output**: Binary classification (sigmoid)
-   **Preprocessing**: EfficientNet preprocessing function

## Troubleshooting

### Error Loading Model

```
Error loading model: [Errno 2] No such file or directory: 'best_model.h5'
```

**Solusi**: Pastikan file `best_model.h5` berada di directory yang sama dengan `app.py`.

### Permission Denied untuk Camera

**Solusi**:

-   Pastikan browser memiliki permission untuk akses camera
-   Gunakan HTTPS untuk deployment production
-   Test dengan browser yang berbeda

### Memory Error

**Solusi**:

-   Reduce batch size dalam kode jika diperlukan
-   Gunakan server dengan RAM yang lebih besar
-   Optimize model dengan quantization

### Slow Prediction

**Solusi**:

-   Gunakan GPU jika tersedia
-   Optimize image preprocessing
-   Reduce image size sebelum processing

## Kustomisasi

### Mengubah Model Path

Edit `app.py` line 18:

```python
MODEL_PATH = 'path/to/your/model.h5'
```

### Mengubah Image Size

Edit preprocessing function di `app.py`:

```python
image = image.resize((224, 224))  # Ubah sesuai kebutuhan
```

### Mengubah Class Labels

Edit `app.py` line 25:

```python
CLASS_LABELS = {0: 'no_helmet', 1: 'has_helmet'}  # Sesuaikan label
```

## Deployment

### Local Development

```bash
python app.py
```

### Production dengan Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment

Buat `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## Lisensi

Bebas digunakan untuk keperluan pembelajaran dan pengembangan.

## Support

Jika ada pertanyaan atau masalah,
