import os
import logging
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import uuid

# =========================
# Reduce TensorFlow Memory Usage
# =========================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

logging.getLogger("absl").setLevel(logging.ERROR)

# Disable GPU (important for Render)
tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)

# =========================
# LOAD MODEL LAZY
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pestguard_cotton_model.h5")

model = None

def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

classes = [
    "Aphids",
    "Armyworm",
    "Healthy",
    "Leaf Curl",
    "Powdery Mildew",
    "Target Spot"
]

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# PREDICT API
# =========================
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    unique_name = str(uuid.uuid4()) + ".jpg"
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)

    file.save(file_path)

    try:
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        model = get_model()

        prediction = model.predict(img_array, verbose=0)
        predicted_index = int(np.argmax(prediction))

        pest_name = classes[predicted_index]
        confidence = float(np.max(prediction))

        return jsonify({
            "prediction": pest_name,
            "confidence": confidence,
            
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# HOME ROUTE
# =========================
@app.route("/")
def home():
    return "PestGuard ML API Running"

# =========================
# LOCAL RUN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
