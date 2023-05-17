import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

app = Flask(__name__, template_folder='template')

model = load_model('keras_model.h5')

classes = ['Tahu tidak berformalin', 'Tahu berformalin']

def preprocess_image(image):
    image = image / 255.0
    image = tf.image.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    return image

def is_tahu(image):
    # Konversi gambar ke mode LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Ambil kanal L dari gambar LAB
    l_channel = lab_image[:, :, 0]

    # Ambil kanal A dan B dari gambar LAB
    ab_channels = lab_image[:, :, 1:]

    # Atur threshold untuk mendeteksi warna putih, kuning, krem, atau coklat
    lower_color = np.array([0, 128], dtype=np.uint8)
    upper_color = np.array([255, 255], dtype=np.uint8)

    # Buat mask untuk warna yang ingin dideteksi
    color_mask = cv2.inRange(ab_channels, lower_color, upper_color)

    # Hitung jumlah piksel non-nol dalam mask
    color_pixel_count = cv2.countNonZero(color_mask)

    # Tentukan threshold untuk mendeteksi tekstur
    texture_threshold = 4

    # Hitung rata-rata perbedaan piksel antara kanal L dan dilasi dari kanal L
    dilated_l_channel = cv2.dilate(l_channel, None, iterations=1)
    texture_diff = cv2.absdiff(l_channel, dilated_l_channel)
    texture_mean = np.mean(texture_diff)

    # Jika gambar memenuhi kriteria warna dan tekstur, anggap itu sebagai gambar tahu
    if color_pixel_count < 3 and texture_mean < texture_threshold:
        return True
    else:
        return False

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    img = Image.open(file.stream)
    img = img.convert("RGB")  # Konversi ke mode RGB

    img = preprocess_image(np.array(img))

    prediction = model.predict(img)

    pred_index = np.argmax(prediction, axis=1)[0]

    if is_tahu(np.array(img[0])):
        result = classes[pred_index]
        prediction_text = "Hasilnya " + result
        return render_template('index.html', prediction_text=prediction_text, label_name=file.filename)
    else:
        return render_template('index.html', prediction_text='Mohon upload gambar tahu', label_name=file.filename)

if __name__ == "__main__":
    app.run(debug=True)
