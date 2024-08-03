# Import libraries
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import keras
import IPython.display as ipd

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv1D, Conv2D, BatchNormalization, MaxPooling1D, MaxPooling2D, Flatten, Dense, Dropout, TimeDistributed, Bidirectional, LSTM, Reshape, UpSampling1D, Lambda, GlobalAveragePooling2D, Conv2DTranspose, Concatenate
from tensorflow.keras.initializers import HeNormal
from sklearn.utils.class_weight import compute_class_weight


def unet(input_shape):
    inputs = tf.keras.Input(input_shape)

    # Contracting path
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=HeNormal(), padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=HeNormal(), padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(0.2)(p1)

    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=HeNormal(), padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=HeNormal(), padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(0.2)(p2)

    # Bottleneck
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=HeNormal(), padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=HeNormal(), padding='same')(c3)
    c3 = BatchNormalization()(c3)

    # Expansive path
    u4 = Conv2DTranspose(128, (2, 2), strides=(2, 2), kernel_initializer=HeNormal(), padding='same')(c3)
    u4 = Concatenate()([u4, c2])
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=HeNormal(), padding='same')(u4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=HeNormal(), padding='same')(c4)
    c4 = BatchNormalization()(c4)

    u5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer=HeNormal(), padding='same')(c4)
    u5 = Concatenate()([u5, c1])
    c5 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=HeNormal(), padding='same')(u5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=HeNormal(), padding='same')(c5)
    c5 = BatchNormalization()(c5)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model