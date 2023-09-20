# %%
import os

from tensorflow import keras
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

# %%
# Set seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
tf.random.set_seed(42)

# %%


def show_training_plots(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_x, test_y):
    loss, accuracy = model.evaluate(test_x, test_y, verbose=0)
    print(f'Test loss: {loss:.3}')
    print(f'Test accuracy: {accuracy:.3}')


# %%  [markdown]
"""
## Excercise 1

### Question 1

- Detected license plates on cars either in a video stream or in a picture.
- In pictures of a street, detect the house numbers.
- Auto checking the answers of a hand written primary school math exam.
- Extracting the amount and bank number from a hand-written check.
- Checking hand written solutions to sudoku puzzles.
"""

# %%
# Load the MNIST dataset
(x_train_raw, y_train_raw), (x_test_raw,
                             y_test_raw) = keras.datasets.mnist.load_data()

# %%
# Check dimesions of the data
print(x_train_raw.shape)
print(x_test_raw.shape)

# %%
# Flatten the data
x_train_flat = x_train_raw.reshape(x_train_raw.shape[0], -1)
x_test_flat = x_test_raw.reshape(x_test_raw.shape[0], -1)

print(x_train_flat.shape)
print(x_test_flat.shape)

# %%
# Rescale the data
x_train_flat = x_train_flat / 255
x_test_flat = x_test_flat / 255

# %%
# Make y categorical
y_train = keras.utils.to_categorical(y_train_raw, 10)
y_test = keras.utils.to_categorical(y_test_raw, 10)

# %%
# Model definition
model = keras.Sequential()
model.add(keras.layers.Dense(256, input_shape=(784,)))
model.add(keras.layers.Dense(10, activation="softmax"))
model.summary()

# %%
# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.RMSprop(),
    metrics='accuracy'
)

# %%
history = model.fit(
    x_train_flat,
    y_train,
    batch_size=128,
    epochs=12,
    verbose=1,
    validation_split=0.2
)

# %%
show_training_plots(history)

# %%
evaluate_model(model, x_test_flat, y_test)

# %%
# Model definition
relu_model = keras.Sequential()
relu_model.add(keras.layers.Dense(256, input_shape=(784,), activation="relu"))
relu_model.add(keras.layers.Dense(10, activation="softmax"))
relu_model.summary()

# %%
# Compile model
relu_model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.RMSprop(),
    metrics='accuracy'
)

# %%
relu_history = relu_model.fit(
    x_train_flat,
    y_train,
    batch_size=128,
    epochs=12,
    verbose=1,
    validation_split=0.2
)

# %%
show_training_plots(relu_history)

# %%
evaluate_model(relu_model, x_test_flat, y_test)

# %%
# Rshape data
x_train_spatial = x_train_raw[..., np.newaxis]
x_test_spatial = x_test_raw[..., np.newaxis]
x_train_spatial = x_train_spatial / 255
x_test_spatial = x_test_spatial / 255

print(x_train_spatial.shape)
print(x_test_spatial.shape)

# %%
cnn_model = keras.Sequential()
cnn_model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                  activation="relu", input_shape=(28, 28, 1)))
cnn_model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                  activation="relu"))
cnn_model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
cnn_model.add(keras.layers.Flatten())
cnn_model.add(keras.layers.Dense(128, activation="relu"))
cnn_model.add(keras.layers.Dense(10, activation="softmax"))

cnn_model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adadelta(learning_rate=1),
    metrics='accuracy'
)

# %%
cnn_history = cnn_model.fit(
    x_train_spatial,
    y_train,
    batch_size=128,
    epochs=6,
    verbose=1,
    validation_split=0.2
)

# %%
show_training_plots(cnn_history)

# %%
evaluate_model(cnn_model, x_test_spatial, y_test)

# %%
dropout_cnn_model = keras.Sequential()
dropout_cnn_model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                          activation="relu", input_shape=(28, 28, 1)))
dropout_cnn_model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                          activation="relu"))
dropout_cnn_model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
dropout_cnn_model.add(keras.layers.Dropout(0.25))
dropout_cnn_model.add(keras.layers.Flatten())
dropout_cnn_model.add(keras.layers.Dense(128, activation="relu"))
dropout_cnn_model.add(keras.layers.Dropout(0.5))
dropout_cnn_model.add(keras.layers.Dense(10, activation="softmax"))

dropout_cnn_model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adadelta(learning_rate=1),
    metrics='accuracy'
)

# %%
dropout_cnn_history = dropout_cnn_model.fit(
    x_train_spatial,
    y_train,
    batch_size=128,
    epochs=6,
    verbose=1,
    validation_split=0.2
)

# %%
show_training_plots(dropout_cnn_history)

# %%
evaluate_model(dropout_cnn_model, x_test_spatial, y_test)

# %%  [markdown]
"""
## Excercise 2
"""

# %%
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# %%
x_train = x_train / 255
x_test = x_test / 255

# %%
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# %%
cifar_cnn = keras.Sequential()
cifar_cnn.add(keras.layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation="relu",
    input_shape=(32, 32, 3),
    padding="same"
))
cifar_cnn.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                  activation="relu"))
cifar_cnn.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
cifar_cnn.add(keras.layers.Dropout(0.25))
cifar_cnn.add(keras.layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation="relu",
))
cifar_cnn.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                  activation="relu"))
cifar_cnn.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
cifar_cnn.add(keras.layers.Dropout(0.25))
cifar_cnn.add(keras.layers.Flatten())
cifar_cnn.add(keras.layers.Dense(512, activation="relu"))
cifar_cnn.add(keras.layers.Dropout(0.5))
cifar_cnn.add(keras.layers.Dense(10, activation="softmax"))

cifar_cnn.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.RMSprop(learning_rate=0.0001, weight_decay=1e-6),
    metrics='accuracy'
)

# %%
cifar_cnn_history = cifar_cnn.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=20,
    verbose=1,
    validation_data=(x_test, y_test),
    shuffle=True
)

# %% 
show_training_plots(cifar_cnn_history)
