# Import libraries
import time
from modal import App, Image, Volume, NetworkFileSystem, wsgi_app
import numpy as np

dockerhub_image = Image.from_registry(
    "tensorflow/tensorflow:2.12.0-gpu",
).pip_install("protobuf==3.20.*")

app = App("train-model", image=dockerhub_image)

vol = Volume.from_name('model_weight')
MODEL_DIR = "/models"
fs = NetworkFileSystem.from_name("training-music", create_if_missing=True)
logdir = "/tensorboard"

@app.function(volumes={MODEL_DIR: vol}, network_file_systems={logdir: fs}, gpu="L4:4", timeout=60000)
def train(train_x, train_y, val_x, val_y):
  import pathlib
  import numpy as np
  import tensorflow as tf
  from tensorflow.keras import layers
  from tensorflow.keras import losses
  from tensorflow.keras import optimizers
  from tensorflow.keras import initializers
  from tensorflow.keras import callbacks

  # def unet(input_shape):
  #   inputs = layers.Input(input_shape)

  #   # Contracting path
  #   c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(inputs)
  #   c1 = layers.BatchNormalization()(c1)
  #   c1 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(c1)
  #   c1 = layers.BatchNormalization()(c1)
  #   p1 = layers.MaxPooling2D((2, 2))(c1)
  #   # p1 = layers.Dropout(0.2)(p1)

  #   c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(p1)
  #   c2 = layers.BatchNormalization()(c2)
  #   c2 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(c2)
  #   c2 = layers.BatchNormalization()(c2)
  #   p2 = layers.MaxPooling2D((2, 2))(c2)
  #   # p2 = layers.Dropout(0.2)(p2)

  #   # Bottleneck
  #   c3 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(p2)
  #   c3 = layers.BatchNormalization()(c3)
  #   c3 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(c3)
  #   c3 = layers.BatchNormalization()(c3)

  #   # Expansive path
  #   u4 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(c3)
  #   u4 = layers.Concatenate()([u4, c2])
  #   c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(u4)
  #   c4 = layers.BatchNormalization()(c4)
  #   c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(c4)
  #   c4 = layers.BatchNormalization()(c4)

  #   u5 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(c4)
  #   u5 = layers.Concatenate()([u5, c1])
  #   c5 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(u5)
  #   c5 = layers.BatchNormalization()(c5)
  #   c5 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(c5)
  #   c5 = layers.BatchNormalization()(c5)

  #   outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)


  #   model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
  #   return model
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
  
  def create_weight_map(y_train, weight_for_1=14, weight_for_0=1):
    weight_map = np.ones(y_train.shape)
    weight_map[y_train == 1] = weight_for_1  # Increase weight for class 1
    weight_map[y_train == 0] = weight_for_0  # Normal weight for class 0
    return weight_map
  
  weight_map = create_weight_map(train_y)
  
  model = unet((16, 128, 1))

  # checkpoint_callback = callbacks.ModelCheckpoint(
  #   filepath='./models/model_checkpoint_epoch_{epoch:02d}.h5',
  #   save_weights_only=False,
  #   save_freq='epoch'
  # )

  reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_mean_squared_error',  # Monitor validation MSE
    factor=0.8,                       # Reduce factor (new_lr = lr * factor)
    patience=3,                      # Number of epochs with no improvement after which learning rate will be reduced
    verbose=1,                        # Int to print messages to stdout
    mode='min',                       # In 'min' mode, lr will reduce when the quantity monitored has stopped decreasing
    min_delta=0.0001,                 # Threshold for measuring the new optimum, to only focus on significant changes
    cooldown=0,                       # Number of epochs to wait before resuming normal operation after lr has been reduced.
    min_lr=0.00001                    # Lower bound on the learning rate
)

  model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss=losses.MeanSquaredError(),
    metrics=[
        tf.keras.metrics.MeanSquaredError(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
    ]
  )

  model.summary()


  tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=1,
  )

  model.fit(
    train_x, train_y,
    sample_weight=weight_map,
    epochs=100,
    callbacks=[reduce_lr, tensorboard_callback],  # Add other callbacks if necessary
    validation_data=(val_x, val_y),
    batch_size=32,
    # class_weight={0: 1, 1: 25}  # Adjust class weights if necessary
  )

  model.save_weights(MODEL_DIR + "/model.h5")

  # with open(MODEL_DIR+"/model.h5", "w") as f:
  #   f.write(model)
  vol.commit()
  # save(MODEL_DIR, model)



@app.function(network_file_systems={logdir: fs})
@wsgi_app()
def tensorboard_app():
    import tensorboard

    board = tensorboard.program.TensorBoard()
    board.configure(logdir=logdir)
    (data_provider, deprecated_multiplexer) = board._make_data_provider()
    wsgi_app = tensorboard.backend.application.TensorBoardWSGIApp(
        board.flags,
        board.plugin_loaders,
        data_provider,
        board.assets_zip_provider,
        deprecated_multiplexer,
    )
    return wsgi_app


@app.local_entrypoint()
def main(just_run: bool = False):
    train_x = np.load('./prepared_data/train_x.npy')
    train_y = np.load('./prepared_data/train_y.npy')
    val_x = np.load('./prepared_data/val_x.npy')
    val_y = np.load('./prepared_data/val_y.npy')

    train.remote(train_x, train_y, val_x, val_y)
    if not just_run:
        print(
            "Training is done, but the app is still running TensorBoard until you hit ctrl-c."
        )
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Terminating app")