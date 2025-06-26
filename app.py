import os
import io
import base64
import logging
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==============================================================================
# 1. Konfigurasi Aplikasi dan Logging
# ==============================================================================

# Siapkan logging
# Di production, Gunicorn akan menangani output ini.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Baca konfigurasi dari environment variables dengan nilai default
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024)) # 16MB
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
MODEL_PATH = os.getenv('MODEL_PATH', 'model/base_model.keras')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Pastikan folder upload ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==============================================================================
# 2. Pemuatan Model
# ==============================================================================

model = None
try:
    # Untuk inferensi, compile=False lebih efisien karena kita tidak perlu optimizer.
    model = load_model(MODEL_PATH, compile=False)
    # Jika Anda tetap ingin compile (misalnya untuk mengevaluasi metrik), ini tidak salah.
    # Namun untuk kecepatan inferensi murni, tidak diperlukan.
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    logging.info(f"Model berhasil dimuat dari path: {MODEL_PATH}")
except Exception as e:
    logging.error(f"Error loading model from {MODEL_PATH}: {e}", exc_info=True)
    # Aplikasi akan tetap berjalan, tetapi endpoint prediksi akan mengembalikan error.

# Class labels
CLASS_LABELS = {0: 'without_helmet', 1: 'with_helmet'}

# ==============================================================================
# 3. Fungsi Helper
# ==============================================================================

def allowed_file(filename):
    """Memeriksa apakah ekstensi file diizinkan."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Melakukan preprocessing gambar untuk input model."""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(image, dtype=np.float32)
        
        # Validasi shape setelah di-array-kan
        if img_array.shape != (224, 224, 3):
            raise ValueError(f"Unexpected image shape after resize: {img_array.shape}")

        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}", exc_info=True)
        return None

def predict_image(img_array: np.ndarray) -> dict:
    """Melakukan prediksi pada gambar yang sudah di-preprocess."""
    if model is None:
        return {"error": "Model is not loaded or failed to load."}
    
    try:
        if img_array.shape != (1, 224, 224, 3):
            return {"error": f"Invalid input shape: {img_array.shape}, expected (1, 224, 224, 3)"}

        prediction = model.predict(img_array, verbose=0)
        confidence = float(prediction[0][0])
        
        predicted_class = 1 if confidence > 0.5 else 0
        
        # Hitung persentase keyakinan yang lebih intuitif
        if predicted_class == 1:
            confidence_percentage = confidence * 100
        else:
            confidence_percentage = (1 - confidence) * 100
        
        result = {
            'predicted_class': predicted_class,
            'class_name': CLASS_LABELS[predicted_class],
            'confidence': round(confidence_percentage, 2),
            'raw_prediction': round(confidence, 4) # Nilai mentah dari sigmoid
        }
        
        logging.info(f"Prediction result: {result}")
        return result
        
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return {"error": "Prediction failed due to an internal server error."}

# ==============================================================================
# 4. Endpoints / Routes
# ==============================================================================

@app.route('/')
def index():
    """Halaman utama untuk demo (jika ada)."""
    return render_template('index.html')

@app.route('/predict/upload', methods=['POST'])
def predict_upload():
    """Endpoint untuk prediksi dari file upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    
    if file and allowed_file(file.filename):
        try:
            image = Image.open(file.stream)
            img_array = preprocess_image(image)
            
            if img_array is None:
                return jsonify({'error': 'Failed to preprocess image'}), 400
            
            result = predict_image(img_array)
            
            if 'error' in result:
                # Jika error berasal dari fungsi predict, berikan status 500
                return jsonify(result), 500
            
            return jsonify(result), 200
            
        except Exception as e:
            logging.error(f"Error processing uploaded file: {e}", exc_info=True)
            return jsonify({'error': 'Error processing image file'}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/predict/camera', methods=['POST'])
def predict_camera():
    """Endpoint untuk prediksi dari data gambar base64."""
    json_data = request.get_json()
    if not json_data or 'image' not in json_data:
        return jsonify({'error': 'No image data provided in JSON payload'}), 400
    
    try:
        # Decode base64 image
        image_data = json_data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        image = Image.open(io.BytesIO(image_bytes))
        img_array = preprocess_image(image)
        
        if img_array is None:
            return jsonify({'error': 'Failed to preprocess image'}), 400
        
        result = predict_image(img_array)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result), 200
        
    except (base64.binascii.Error, IndexError) as e:
        logging.warning(f"Invalid base64 data received: {e}")
        return jsonify({'error': 'Invalid base64 image data'}), 400
    except Exception as e:
        logging.error(f"Error processing camera image: {e}", exc_info=True)
        return jsonify({'error': 'Error processing camera image'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint untuk monitoring."""
    model_status = "loaded" if model is not None else "not_loaded"
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'message': 'Helmet Detection API is running'
    })

# Blok if __name__ == '__main__' tidak akan dijalankan oleh Gunicorn.
# Ini tetap berguna untuk menjalankan server development secara lokal.
if __name__ == '__main__':
    # Pastikan untuk TIDAK menggunakan debug=True di production!
    # Gunakan Gunicorn untuk production.
    # Baris ini hanya untuk kemudahan testing lokal.
    app.run(host='0.0.0.0', port=5000, debug=False)