import tensorflow as tf
import numpy as np
import io
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
from tensorflow.keras import callbacks

def convnet():
  input_layer = layers.Input(shape=(5, 229, 1))
  # Layer 1
  x1 = layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer=tf.keras.initializers.HeNormal(), padding="valid")(input_layer)
  x2 = layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer=tf.keras.initializers.HeNormal(), padding="valid")(x1)
  x3 = layers.BatchNormalization()(x2)
  x4 = layers.MaxPool2D((1, 2))(x3)
  x5 = layers.Dropout(0.25)(x4)

  # Layer 2
  x6 = layers.Conv2D(32, (1, 3), activation="relu", kernel_initializer=tf.keras.initializers.HeNormal(), padding="valid")(x5)
  x7 = layers.BatchNormalization()(x6)
  x8 = layers.Conv2D(32, (1, 3), activation="relu", kernel_initializer=tf.keras.initializers.HeNormal(), padding="valid")(x7)
  x9 = layers.BatchNormalization()(x8)
  x10 = layers.MaxPool2D((1, 2))(x9)
  x11 = layers.Dropout(0.25)(x10)

  # Layer 3
  x12 = layers.Conv2D(64, (1, 25), activation="relu", kernel_initializer=tf.keras.initializers.HeNormal(), padding="valid")(x11)
  x13 = layers.BatchNormalization()(x12)

  # Layer 4
  x14 = layers.Conv2D(128, (1, 25), activation="relu", kernel_initializer=tf.keras.initializers.HeNormal(), padding="valid")(x13)
  x15 = layers.BatchNormalization()(x14)
  x16 = layers.Dropout(0.5)(x15)
  x17 = layers.Conv2D(128, (1, 1), activation="relu", kernel_initializer=tf.keras.initializers.HeNormal(), padding="valid")(x16)
  x18 = layers.BatchNormalization()(x17)
  x19 = layers.AveragePooling2D((1, 6))(x18)
  outputs = layers.Activation("sigmoid")(x19)

  model = tf.keras.Model(inputs=[input_layer], outputs=[outputs])
  return model