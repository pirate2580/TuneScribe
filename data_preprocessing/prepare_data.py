"""
This file prepares data into (None, 8, 128, 1) data and (None 8, 128, 1) labels into npy files to
feed to the deep learning model
"""
from process_data import generate_x, generate_y
import numpy as np
import os
import pickle


if __name__ == '__main__':

  

  with open(os.path.join('./pickle_data/training_data.pkl'), 'rb') as file:
    unprocessed_training_data = pickle.load(file)
  
  with open(os.path.join('./pickle_data/training_label.pkl'), 'rb') as file:
    unprocessed_training_label = pickle.load(file)
  
  with open(os.path.join('./pickle_data/validation_data.pkl'), 'rb') as file:
    unprocessed_validation_data = pickle.load(file)

  with open(os.path.join('./pickle_data/validation_label.pkl'), 'rb') as file:
    unprocessed_validation_label = pickle.load(file)
  
  train_x, train_y, val_x, val_y = [], [], [], []

  for song_id in unprocessed_training_data:
    x, padding = unprocessed_training_data[song_id][0], unprocessed_training_data[song_id][1]
    padding = padding // 2048 + 1
    y = unprocessed_training_label[song_id]
    for j in range(0, 6460 - padding, 16):
      train_x.append(x[:, j:j+16])
      train_y.append(y[j:j+16, :])

  for song_id in unprocessed_validation_data:
    x, padding = unprocessed_validation_data[song_id][0], unprocessed_validation_data[song_id][1]
    padding = padding // 2048 + 1
    y = unprocessed_validation_label[song_id]
    for j in range(0, 6460 - padding, 16):
      val_x.append(x[:, j:j+16])
      val_y.append(y[j:j+16, :])
  
  train_x = np.array(train_x)
  train_y = np.array(train_y)
  val_x = np.array(val_x)
  val_y = np.array(val_y)

  train_x = train_x.reshape(-1, 16, 128, 1)
  val_x = val_x.reshape(-1, 16, 128, 1)
  train_y = train_y.reshape(-1, 16, 128, 1)
  val_y = val_y.reshape(-1, 16, 128, 1)
  print()
  print(train_x.shape)
  print(train_y.shape)
  print(val_x.shape)
  print(val_y.shape)

  np.save('./prepared_data/train_x.npy', train_x)
  np.save('./prepared_data/train_y.npy', train_y)
  np.save('./prepared_data/val_x.npy', val_x)
  np.save('./prepared_data/val_y.npy', val_y)
    
  # # print(unprocessed_training_data)
  