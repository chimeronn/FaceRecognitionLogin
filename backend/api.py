from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
from FaceModel import FaceModel
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = FaceModel(np.empty((0, 512*512)), 0, 512*512, np.empty(0))
model.load(os.path.join(BASE_DIR, "face_model.npz"))

def crop_center(img, num_pixels):
    w, h = img.size
    left = (w - 2 * num_pixels) // 2
    top = (h - 2 * num_pixels) // 2
    right = left + 2 * num_pixels
    bottom = top + 2 * num_pixels
    return img.crop((left, top, right, bottom))

def preprocess_img(url):
    header, encoded_img = url.split(",", 1)
    img_bytes = base64.b64decode(encoded_img)
    img = Image.open(BytesIO(img_bytes)).convert("L")
    img = img.resize((512, 512))
    img = crop_center(img, 256)
    img_array = np.array(img).astype(np.float32).flatten() / 255.0
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input = preprocess_img(data["image"])
    res = model.predict(input)
    return jsonify({"result": "My Face" if res else "Unknown"})

if __name__ == "__main__":
    app.run(debug=True, port=8080)