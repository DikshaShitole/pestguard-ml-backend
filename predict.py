import os
import sys
import logging

# Suppress TensorFlow & absl warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("absl").setLevel(logging.ERROR)

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# =========================
# CHECK IMAGE PATH
# =========================
if len(sys.argv) < 2:
    print("NO_IMAGE_PATH")
    sys.exit(1)

img_path = sys.argv[1]

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "pestguard_cotton_model (1).h5")
    )
except Exception as e:
    print("MODEL_LOAD_ERROR")
    sys.exit(1)

# =========================
# IMAGE PREPROCESSING
# =========================
try:
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
except:
    print("IMAGE_LOAD_ERROR")
    sys.exit(1)

# =========================
# PREDICTION
# =========================
prediction = model.predict(img_array, verbose=0)

predicted_index = int(np.argmax(prediction))
num_outputs = prediction.shape[1]

# 🔥 MUST MATCH TRAINING CLASSES (6)
classes = [
    "Aphids",
    "Armyworm",
    "Healthy",
    "Leaf Curl",
    "Powdery Mildew",
    "Target Spot"
]

# =========================
# SAFETY CHECK
# =========================
if num_outputs != len(classes):
    print("CLASS_MISMATCH")
    sys.exit(1)

# =========================
# FINAL OUTPUT (ONLY PEST NAME)
# =========================
print(classes[predicted_index])
