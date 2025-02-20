"""
Helper function that converts audio into wav and preprocesses it for the model
"""

import librosa
from pydub import AudioSegment
import io
import numpy as np

def preprocess_audio(file_stream):
    """
    Preprocesses the audio file for prediction.
    1. Converts any audio format to WAV using pydub.
    2. Loads the WAV data into librosa.
    3. Extracts mel spectrogram features.
    4. Returns the features along with the original audio duration (in seconds).
    """

    # Read the file from file_stream
    audio_bytes = file_stream.read()

    # Convert the incoming audio to WAV using pydub
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    except Exception as e:
        print("Error reading audio file. Make sure ffmpeg is installed for non-WAV formats.")
        raise e

    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format='wav')  # Export as WAV
    wav_buffer.seek(0)  # Go back to the start of the BytesIO buffer

    print("Loading audio file with librosa...")
    # Load the WAV data at a specific sample rate (11025 in this case)
    audio_data, sr = librosa.load(wav_buffer, sr=11025)
    print(f"Loaded audio data with length: {len(audio_data)} and sample rate: {sr}")

    # Pad audio to a fixed length (here, 1200 seconds or 20 minutes if sr=11025)
    desired_length = 1200 * sr  # e.g., 1200 seconds
    if len(audio_data) < desired_length:
        padding_length = desired_length - len(audio_data)
        padded_audio_data = np.pad(audio_data, (0, padding_length), 'constant')
    else:
        # If the audio is longer, you might decide to truncate or handle differently
        padded_audio_data = audio_data[:desired_length]
        padding_length = 0

    print(f"Audio padded/truncated to length {len(padded_audio_data)}")

    # Convert audio to a mel spectrogram
    try:
        mel_spectrogram = librosa.feature.melspectrogram(
            y=padded_audio_data,
            sr=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=229
        )
        print(f"Mel spectrogram shape: {mel_spectrogram.shape}")
    except Exception as e:
        print(f"Error creating mel spectrogram: {e}")
        raise

    # Build the input data windowed in steps of 5 frames
    input_data = []
    # Each hop in your iteration is 5 frames along the time axis
    step_frames = 5
    # The number of time frames to skip if there's leftover padding
    time_padding = padding_length // 512 + 1

    for j in range(0, 25840 - time_padding, step_frames):
        input_data.append(mel_spectrogram[:, j : j + step_frames])

    input_data = np.array(input_data)
    # Reshape to match (batch, time, frequency, channels)
    # as expected by your deep learning model
    input_data = np.reshape(
        input_data, 
        (input_data.shape[0], input_data.shape[2], input_data.shape[1], 1)
    )
    print(f"Final input shape: {input_data.shape}")

    # Return the processed data and the original duration in seconds
    return input_data, len(audio_data) // sr