from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import io
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
from tensorflow.keras import callbacks

from model import convnet
from preprocess_audio import preprocess_audio

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

model = convnet()
model.load_weights("cnn_model.h5")



@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 402

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 401

    try:
        # Preprocess the audio file
        input_data, audio_length = preprocess_audio(file)
        print('preprocessed all data')
        # Make predictions using the model
        predictions = model.predict(input_data)

        print(f"prediction shape is NOW: {predictions.shape}")

        predictions = predictions.reshape(-1, 128)
        print(f"prediction shape is NOW: {predictions.shape}")
        # Set a threshold value
        threshold = 0.5

        # Apply thresholding
        binary_y = (predictions > threshold).astype(int)
        # print(binary_y.shape)
        binary_list = binary_y.tolist()

        print("Midi has been loaded")

        return jsonify({"midi_array": binary_list}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)