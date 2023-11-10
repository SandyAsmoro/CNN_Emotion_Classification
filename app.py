from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)
emotion_model = load_model('emotion_model.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Ambil gambar dari HTML
        image = request.files['image'].read()
        npimg = np.fromstring(image, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))
        img = img.reshape(1, 48, 48, 1) / 255.0

        # Lakukan prediksi
        prediction = emotion_model.predict(img)
        predicted_emotion = np.argmax(prediction)

        # Konversi hasil prediksi ke string
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        predicted_emotion_str = emotion_labels[predicted_emotion]

        return render_template('result.html', emotion=predicted_emotion_str)

if __name__ == '__main__':
    app.run(debug=True)
