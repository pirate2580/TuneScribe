# TuneScribe

TuneScribe is a web-based application designed to transform audio files from WAV format into MIDI sequences. The application leverages advanced deep learning techniques, employing a U-Net architecture to analyze audio spectrograms and predict corresponding MIDI events.

## Features

- **WAV to MIDI Conversion**: Users can upload WAV files and convert them into MIDI format.
- **Sheet Music Visualization**: Convert MIDI files into sheet music using VexFlow, directly visible within the web interface.
- **Interactive UI**: Upload, convert, and download capabilities alongside playback controls for MIDI files.

## Technical Overview

### Audio Processing

The core of TuneScribe's functionality lies in its ability to process audio data through a series of deep learning models. Here's a breakdown of the audio processing steps:

#### Spectrogram Conversion

- **Audio Loading**: Audio files are loaded and resampled to a consistent sampling rate to ensure uniformity in input data.
- **Spectrogram Generation**: The application converts audio waveforms into Mel spectrograms. Spectrograms provide a time-frequency representation of the audio, which is crucial for the neural network to understand and process audio data effectively.

  Spectrograms are generated using `librosa`, a Python library for music and audio analysis. Parameters such as `n_fft`, `hop_length`, and `n_mels` are tuned to optimize the extraction of relevant features from the audio.

#### Neural Network Prediction

- **U-Net Architecture**: The project uses a U-Net model, a type of convolutional neural network that is especially effective for tasks where context and localization are important, such as segmenting musical notes from spectrograms.
- **Training**: The model is trained on a dataset of paired audio files and their corresponding MIDI files, allowing it to learn the mapping from audio spectrogram to MIDI data.

### MIDI Conversion

- **Thresholding and Post-Processing**: The network output is thresholded to binary values representing note activations. This binary matrix is then converted into a series of MIDI messages, specifying note onsets and offsets.
- **MIDI File Generation**: Using the `mido` library, MIDI messages are compiled into a standard MIDI file that can be played or used in other music software.

## Installation

Follow these steps to set up and run TuneScribe:

### Backend

```bash
# Clone the repository and navigate into it
git clone https://github.com/yourusername/tunescribe.git
cd tunescribe

# Set up a Python virtual environment and activate it
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install Python dependencies
pip install -r requirements.txt

# Run the Flask server
python server.py

### Frontend

# Navigate to the frontend directory from the root of the project
cd client/frontend

# Install dependencies
npm install

# Start the React application
npm start
