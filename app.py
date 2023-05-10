import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__, template_folder='template')

model = load_model('keras_model.h5')

classes = ['Tahu tidak berformalin', 'Tahu berformalin']

def preprocess_image(image):
    image = image / 255.0
    image = tf.image.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    img = Image.open(file.stream)

    img = preprocess_image(np.array(img))

    prediction = model.predict(img)

    pred_index = np.argmax(prediction, axis=1)[0]

    result = classes[pred_index]
    prediction_text = "Hasilnya " + result
    return render_template('index.html', prediction_text=prediction_text, label_name=file.filename)

if __name__ == "__main__":
    app.run(debug=True)