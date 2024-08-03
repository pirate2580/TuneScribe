"""
This file contains helper functions to preprocess training/validation data and labels
"""

# Imports
import numpy as np
import pandas as pd
import os
import librosa

def generate_x(file_path: str, sample_rate: int = 11025, max_time: int = 1200, n_fft: int = 4096, hop_length: int = 2048, n_mels: int = 128) -> dict:
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
    padding_length = max_time * sr - len(audio_data)              # padding data to 20 minutes
    padded_audio_data = np.pad(audio_data, (0, padding_length), 'constant')
    mel_spectrogram = librosa.feature.melspectrogram(y=padded_audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels) # creating a mel spectrogram of the song data, 4096 window size with 11025 sample rate is ~0.33 seconds of audio data
    x_data[wav_file[0:len(wav_file) - 4]] = mel_spectrogram

    print(f"song {wav_file} has been processed")
  
  return x_data

def generate_y(file_path: str, max_time: int = 1200) -> dict:
  """Function to generate ouput labels for either training or validation data"""
  new_time_steps = 6460
  downsample_factor = max_time * 44100 / new_time_steps

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

      start_time = round(start_time / downsample_factor, 1)       # downsample time, losing temporal resolution, to fit labels to data
      end_time = round(end_time / downsample_factor, 1)
      audio_label[int(start_time):int(end_time), int(note)-20] = 1
    print(f"label for song {csv_file} is processed")
    y_data[csv_file[0:len(csv_file) - 4]] = audio_label
  
  return y_data


if __name__ == '__main__':

  training_data = generate_x('./raw_data/musicnet/train_data')
  training_label = generate_y('./raw_data/musicnet/train_labels')
  print(training_data['2177'].shape)
  print(training_label['2177'].shape)



# # List all files in the directory
# files = os.listdir('./musicnet/train_data')
# # Filter the list to include only CSV files
# wav_files = [f for f in files if f.endswith('.wav')]


# for i in range (len(wav_files)):
#   wav_file = wav_files[i]
#   file_path = os.path.join('./musicnet/train_data', wav_file)
#   audio_data, sr = librosa.load(file_path, sr=sample_rate)  # audio song processed at 11025 Hz
#   # print(audio_data.shape)
#   padding_length = max_time * sr - len(audio_data)
#   # print(padding_length)

#   pickle_file_path = os.path.join('./musicnet_pickle', wav_file[0:len(wav_file) - 3] + 'pkl')
#   padded_audio_data = np.pad(audio_data, (0, padding_length), 'constant')
#   # print(padded_audio_data.shape)
#   mel_spectrogram = librosa.feature.melspectrogram(y=padded_audio_data, sr=sr, n_fft=4096, hop_length=2048, n_mels=88)
#   # print(mel_spectrogram.shape)
#   with open(pickle_file_path, 'wb') as f:
#       pickle.dump((mel_spectrogram, padding_length), f)
#   print(i)