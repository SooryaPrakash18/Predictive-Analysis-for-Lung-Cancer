# app.py

from flask import Flask, render_template, request, send_from_directory
import numpy as np
import pickle
from PIL import Image, UnidentifiedImageError
import os

app = Flask(__name__)

MODEL_PATH = "cancer_detection_model.pkl"
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

IMAGE_SIZE = (50, 50)
CATEGORIES = ["cancer", "non_cancer"]

UPLOAD_FOLDER = "temp"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

DATASET_PATH = r"D:\\MCA\\Final\\Project\\myenv\\project\\dataset"

@app.route("/temp/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", error="No file uploaded!")

        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", error="No file selected!")

        try:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            img = Image.open(filepath).convert("L").resize(IMAGE_SIZE)
            img_category = None

            for category in CATEGORIES:
                folder_path = os.path.join(DATASET_PATH, category)
                if file.filename in os.listdir(folder_path):
                    img_category = category
                    break

            if img_category is None:
                return render_template("index.html", error="Unknown Image: It is not related to cancer.")

            img_array = np.array(img).flatten().reshape(1, -1)
            prediction = model.predict(img_array)[0]
            confidence = model.predict_proba(img_array)[0][prediction]

            if confidence < 0.6:
                result = "Unknown Image"
                color = "secondary"
            else:
                result = "Cancer Detected" if prediction == 0 else "Non-Cancer Detected"
                color = "danger" if prediction == 0 else "success"

            return render_template("index.html", result=result, confidence=confidence, color=color, image_path=f"/temp/{file.filename}")

        except UnidentifiedImageError:
            return render_template("index.html", error="Error: Uploaded file is not a valid image.")
        except Exception as e:
            return render_template("index.html", error=f"Unknown Error: {e}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
