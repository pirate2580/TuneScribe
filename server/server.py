from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import librosa
import io

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('./model_weights/model.h5')

def preprocess_audio(file_stream):
    """
    Preprocesses the audio file for prediction.
    Load the wav file, extract features and return them.
    """
    # Load the audio file as a waveform
    audio_data, sr = librosa.load(io.BytesIO(file_stream.read()), sr=11025)
    padding_length = 1200 * sr - len(audio_data)
    padded_audio_data = np.pad(audio_data, (0, padding_length), 'constant')

    # Convert audio to a mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(audio_data, sr=sr, n_mels=128)

    input_data= []
    padding_length = padding_length// 2048 + 1
    for j in range(0, 6460 - padding_length, 16):
      input_data.append(input_data[:, j:j+16])
    input_data = np.array(input_data)
    input_data = np.reshape(input_data, (input_data.shape[0], input_data.shape[2], input_data.shape[1], 1))
    return input_data

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Preprocess the audio file
        input_data = preprocess_audio(file)

        # Make predictions using the model
        predictions = model.predict(input_data)

        # Convert predictions to a list and return as JSON
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
