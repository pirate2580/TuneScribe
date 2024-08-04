# Import libraries
import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
import IPython.display as ipd
import wave
import pickle
from mido import Message, MidiFile, MidiTrack


if __name__ == '__main__':
  train_data = np.load('./prepared_data/train_x.npy')
  train_label = np.load('./prepared_data/train_y.npy')
  val_data = np.load('./prepared_data/val_x.npy')
  val_label = np.load('./prepared_data/val_y.npy')
  print(train_data.shape)
  print(train_label.shape)
  print(val_data.shape)
  print(val_label.shape)
  # with open(os.path.join('./pickle_data/training_label.pkl'), 'rb') as file:
  #   data = pickle.load(file)  # og shape (n_mels = 88, 51680 timesteps)

  # song = data['2618']

  # # Example numpy array with shape (12920, 88)
  # # 1 represents a note on event, 0 represents a note off event
  # np.random.seed(0)  # For reproducibility
  # # timesteps = 25840
  # timesteps = 6460
  # notes = 128

  # # Create a new MIDI file and a track
  # mid = MidiFile()
  # track = MidiTrack()
  # mid.tracks.append(track)

  # # Define constants
  # ticks_per_beat = 480  # MIDI resolution
  # total_seconds = 484 
  # total_ticks = ticks_per_beat * 4 * total_seconds  # Assuming 4/4 time signature

  # # Calculate time per timestep in ticks
  # time_per_timestep = total_ticks / timesteps
  # print(time_per_timestep)
  # # Initialize current note states (to track note on/off)
  # current_note_states = [0] * notes

  # # Iterate over each timestep
  # for timestep in range(timesteps):
  #   for note in range(notes):
  #     state = song[timestep, note]
  #     if state != current_note_states[note]:
  #       # If state changed, create a note on/off message
  #       velocity = 64 if state == 1 else 0
  #       message_type = 'note_on' if state == 1 else 'note_off'
  #       track.append(Message(message_type, note=note + 21, velocity=velocity, time=int(round(time_per_timestep, 1))))
  #       current_note_states[note] = state

  # # Save the MIDI file
  # mid.save('./midi_testing/output.mid')