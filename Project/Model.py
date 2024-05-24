from flask import Flask, request, jsonify
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.models import load_model

app = Flask(__name__)
model = load_model("my_model.h5")


@app.route("/sendphoto", methods=["POST"])
def sendphoto():
    if request.method == "POST":
        base64_str = request.json.get("image")
        img_bytes = base64.b64decode(base64_str)

        newimg = Image.open(BytesIO(img_bytes))
        newimg = newimg.convert("RGB")

        newimg = newimg.resize((512, 512))
        img_array = image.img_to_array(newimg)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)

        return jsonify({"predicted_class": int(predicted_class)})


@app.route("/take_photo", methods=["POST"])
def take_photo():
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            img = Image.open(file)
            img = img.convert("RGB")
            img = img.resize((512, 512))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)
            return jsonify({"predicted_class": int(predicted_class)})
        else:
            return jsonify({"error": "No image file provided"}), 400


@app.route("/get_info", methods=["GET"])
def get_info():
    if request.method == "GET":
        data = {
            "status": "online"
        }
        return jsonify(data)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
