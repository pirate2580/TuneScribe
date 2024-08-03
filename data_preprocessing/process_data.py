"""
This file contains helper functions to preprocess training/validation data and labels
"""

# Imports
import numpy as np
import pandas as pd
import os
import librosa
import pickle

def generate_x(file_path: str, sample_rate: int = 11025, n_fft: int = 4096, max_time: int = 1200, hop_length: int = 2048, n_mels: int = 128) -> dict:
  """
    Generate Mel spectrogram data from WAV files in a given directory.
    
    Args:
    - file_path (str): Path to the directory containing WAV files.
    - sample_rate (int): Sample rate for audio processing. Default is 11025 Hz.
    - max_time (int): Maximum length of audio in seconds for padding. Default is 20 minutes.
    - n_fft (int): Length of the FFT window. Default is 4096.
    - hop_length (int): Number of samples between successive frames. Default is 2048.
    - n_mels (int): Number of Mel bands to generate. Default is 128.
    
    Returns:
    - dict: A dictionary with file names as keys and their corresponding Mel spectrograms as values.
  """
  files = os.listdir(file_path)
  wav_files = [f for f in files if f.endswith('.wav')]

  x_data = {}

  for i, wav_file in enumerate(wav_files):
    wav_file_path = os.path.join(file_path, wav_file)

    audio_data, sr = librosa.load(wav_file_path, sr=sample_rate)  # audio song loaded and processed at 11025 Hz
    padding_length = max_time * sr - len(audio_data)              # padding data to 20 minutes unless another time (in secs) is specified
    padded_audio_data = np.pad(audio_data, (0, padding_length), 'constant')
    mel_spectrogram = librosa.feature.melspectrogram(y=padded_audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels) # creating a mel spectrogram of the song data, 4096 window size with 11025 sample rate is ~0.33 seconds of audio data
    # print(mel_spectrogram.shape)
    x_data[wav_file[0:len(wav_file) - 4]] = [mel_spectrogram, padding_length]

    print(f"song {wav_file} has been processed")
  
  return x_data

def generate_y(file_path: str, max_time: int = 1200) -> dict:
  """Function to generate ouput labels for either training or validation data"""
  new_time_steps = 6460                                 # hardcoded, should be .shape[1] of any np arrays from x_data
  downsample_factor = max_time * 44100 / new_time_steps # original time from labels csv file is in frame number where sampled at 44.1 kHz, need downsampling

  files = os.listdir(file_path)
  csv_files = [f for f in files if f.endswith('.csv')]

  y_data = {}

  for i in range(len(csv_files)):
    csv_file = csv_files[i]
    csv_file_path = os.path.join(file_path, csv_file)

    audio_label = np.zeros((new_time_steps, 128))  # padded to 20 minutes, each row represents a timestep, and each of the 128 columns represents if a note is pressed or not at a certain timestep
    df = pd.read_csv(csv_file_path)
    for index, row in df.iterrows():
      start_time, end_time, note = row['start_time'], row['end_time'], row['note']

      start_time = int(start_time) // downsample_factor   # when downsampled a lot like this, there will be rounding errors and won't sound like original exactly :(
      end_time = int(end_time) // downsample_factor
      audio_label[int(start_time):int(end_time), int(note)] = 1

    print(f"label for song {csv_file} is processed")
    y_data[csv_file[0:len(csv_file) - 4]] = audio_label
  
  return y_data


if __name__ == '__main__':

  # main block generates data/labels and saves to pickle files to be prepared in np arrays for distributed training on AWS ec2

  training_data = generate_x('./raw_data/musicnet/train_data')
  training_label = generate_y('./raw_data/musicnet/train_labels')
  validation_data = generate_x('./raw_data/musicnet/test_data')
  validation_label = generate_y('./raw_data/musicnet/test_labels')

  with open('./pickle_data/training_data.pkl', 'wb') as f:
    pickle.dump(training_data, f)

  with open('./pickle_data/training_label.pkl', 'wb') as f:
    pickle.dump(training_label, f)

  with open('./pickle_data/validation_data.pkl', 'wb') as f:
    pickle.dump(validation_data, f)

  with open('./pickle_data/validation_label.pkl', 'wb') as f:
    pickle.dump(validation_label, f)
#   # print(training_data['2177'])