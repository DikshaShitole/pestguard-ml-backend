import os
import logging
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import uuid

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("absl").setLevel(logging.ERROR)

app = Flask(__name__)

# =========================
# LOAD MODEL ON START
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pestguard_cotton_model.h5")

model = tf.keras.models.load_model(MODEL_PATH)

classes = [
    "Aphids",
    "Armyworm",
    "Healthy",
    "Leaf Curl",
    "Powdery Mildew",
    "Target Spot"
]

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# PREDICT API
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Generate unique filename
    unique_name = str(uuid.uuid4()) + ".jpg"
    file_path = os.path.join(UPLOAD_FOLDER, unique_name)

    file.save(file_path)

    try:
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array, verbose=0)
        predicted_index = int(np.argmax(prediction))
        pest_name = classes[predicted_index]

        os.remove(file_path)  # delete after use

        return jsonify({"pest": pest_name})

    except Exception:
        return jsonify({"error": "Prediction failed"}), 500


# =========================
# HOME ROUTE
# =========================
@app.route("/")
def home():
    return "PestGuard ML API Running"


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
