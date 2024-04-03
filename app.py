from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load pre-trained machine learning model
model = tf.keras.models.load_model("path_to_your_model")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    uploaded_files = request.files.getlist("file")
    results = []

    for file in uploaded_files:
        img = Image.open(file).resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        results.append(predictions.tolist()[0])  # Convert predictions to a list and append to results

    return render_template('results.html', results=results)


