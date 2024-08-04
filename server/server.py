from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import librosa
import io

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
from tensorflow.keras import callbacks

from mido import Message, MidiFile, MidiTrack

# def unet(input_shape):
#     inputs = layers.Input(input_shape)

#     # Contracting path
#     c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(inputs)
#     c1 = layers.BatchNormalization()(c1)
#     c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(c1)
#     c1 = layers.BatchNormalization()(c1)
#     p1 = layers.MaxPooling2D((2, 2))(c1)
#     # p1 = layers.Dropout(0.2)(p1)

#     c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(p1)
#     c2 = layers.BatchNormalization()(c2)
#     c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(c2)
#     c2 = layers.BatchNormalization()(c2)
#     p2 = layers.MaxPooling2D((2, 2))(c2)
#     # p2 = layers.Dropout(0.2)(p2)

#     # Bottleneck
#     c3 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(p2)
#     c3 = layers.BatchNormalization()(c3)
#     c3 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(c3)
#     c3 = layers.BatchNormalization()(c3)

#     # Expansive path
#     u4 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(c3)
#     u4 = layers.Concatenate()([u4, c2])
#     c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(u4)
#     c4 = layers.BatchNormalization()(c4)
#     c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(c4)
#     c4 = layers.BatchNormalization()(c4)

#     u5 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(c4)
#     u5 = layers.Concatenate()([u5, c1])
#     c5 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(u5)
#     c5 = layers.BatchNormalization()(c5)
#     c5 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(c5)
#     c5 = layers.BatchNormalization()(c5)

#     outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)


#     model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
#     return model

def unet(input_shape):
    inputs = layers.Input(input_shape)

    # Contracting path
    c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)

    # Expansive path
    u5 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(c4)
    u5 = layers.Concatenate()([u5, c3])
    c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(u5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)

    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(c5)
    u6 = layers.Concatenate()([u6, c2])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(u6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(c6)
    u7 = layers.Concatenate()([u7, c1])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(c7)
    c7 = layers.BatchNormalization()(c7)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model


# Load the model weights
model = unet((16, 128, 1))
model.load_weights('./model_weights/new_model.h5')

app = Flask(__name__)
CORS(app)

def preprocess_audio(file_stream):
    """
    Preprocesses the audio file for prediction.
    Load the wav file, extract features and return them.
    """
    # Load the audio file as a waveform
    print("Loading audio file...")
    audio_data, sr = librosa.load(io.BytesIO(file_stream.read()), sr=11025)
    print(f"Loaded audio data with length: {len(audio_data)} and sample rate: {sr}")

    padding_length = 1200 * sr - len(audio_data)
    padded_audio_data = np.pad(audio_data, (0, padding_length), 'constant')
    print(f"Audio padded to length {len(padded_audio_data)}")

    # Convert audio to a mel spectrogram
    # mel_spectrogram = librosa.feature.melspectrogram(padded_audio_data, sr=sr, n_fft=4096, hop_length=2048, n_mels=128)
    # print(f"Mel spectrogram shape: {mel_spectrogram.shape}")

    try:
        mel_spectrogram = librosa.feature.melspectrogram(y=padded_audio_data, sr=sr, n_fft=4096, hop_length=2048, n_mels=128)
        print(f"Mel spectrogram shape: {mel_spectrogram.shape}")
    except Exception as e:
        print(f"Error creating mel spectrogram: {e}")
        raise

    input_data= []
    padding_length = padding_length// 2048 + 1
    for j in range(0, 6460 - padding_length, 16):
      input_data.append(mel_spectrogram[:, j:j+16])
    input_data = np.array(input_data)
    input_data = np.reshape(input_data, (input_data.shape[0], input_data.shape[2], input_data.shape[1], 1))
    print(input_data.shape)
    return input_data, len(audio_data) // sr

def create_midi_file(binary_y, audio_length):
    np.random.seed(0)  # For reproducibility
    # timesteps = 25840
    timesteps = binary_y.shape[0]
    notes = 128
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    ticks_per_beat = 120
    total_seconds = audio_length
    total_ticks = ticks_per_beat * 4 * total_seconds  # Assuming 4/4 time signature
    time_per_timestep = total_ticks / timesteps
    current_note_states = [0] * notes

    print(f"Total seconds: {total_seconds}, Total ticks: {total_ticks}, Time per timestep: {time_per_timestep}")
    print(binary_y.shape)
    for timestep in range(timesteps):
        for note in range(notes):
            state = binary_y[timestep, note]
            # if (state==1):
            # print(state)
            if state != current_note_states[note]:
                # If state changed, create a note on/off message
                velocity = 64 if state == 1 else 0
                message_type = 'note_on' if state == 1 else 'note_off'
                track.append(Message(message_type, note=note + 21, velocity=velocity, time=int(time_per_timestep)))
                current_note_states[note] = state
    print(f"MIDI file created with {len(track)} messages")
    return mid
    

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Preprocess the audio file
        input_data, audio_length = preprocess_audio(file)
        print('preprocessed all data')
        # Make predictions using the model
        predictions = model.predict(input_data)

        predictions = predictions.reshape(-1, 128)
        print(f"prediction shape is NOW: {predictions.shape}")
        # Set a threshold value
        threshold = 0.98

        # Apply thresholding
        binary_y = (predictions > threshold).astype(int)
        
        number_of_ones = np.sum(binary_y == 1)
        print(number_of_ones)

        predicted_midi_file = create_midi_file(binary_y, audio_length)
        print("Midi has been loaded")

        try:
            byte_io = io.BytesIO()
            predicted_midi_file.save(file=byte_io)
            byte_io.seek(0)
            print("MIDI has been saved, ready to send to front end")
        except Exception as e:
            print(f"Error saving MIDI to byte stream: {e}")
            raise

        return send_file(
            byte_io,
            as_attachment=True,
            download_name='output.mid',
            mimetype='audio/midi'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
