# %%
import os

from tensorflow import keras
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

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

indices = np.arange(len(x_train_raw))
eights = indices[y_train_raw == 8]
selected_index = np.take(eights, 1)
eight = x_train_raw[selected_index]
plt.imshow(eight, cmap="gray")
plt.show()


# %% 
import matplotlib.image

matplotlib.image.imsave('eight.png', eight)

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
y_pred = model.predict(x_test_flat)
y_pred_classes = np.argmax(y_pred, axis=1)

# Compute the Confusion Matrix
conf_matrix = confusion_matrix(y_test_raw, y_pred_classes)

# Plot the Confusion Matrix

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# %%  [markdown]
"""
### Question 2

1. At the first epoch, we can see that the training accuracy is lower than the validation accuracy, and conversely, the training loss is higher than the validation loss.
This might be due to the fact that the validation set is much smaller, thus has less varation within the dataset to be captured by the model. Since the training set is larger, there's a bigger chance that edge cases occur in it, that might be misclassified when the model hasn't training much yet.
2. Then we see that the training accuracy increases, and the training loss decreases rapidly in the first epochs. The validation loss also decreases, and the accuracy also increases, but not nearly as quickly.
This might be due to the fact that the model learns the features from the training set.  These features might not initially generalise too well to the validation set, but instantly perform way better on the images on which is was trained. In a way it tends towards overfitting to the training set.
3. Then we see that the training accuracy keeps on increases, and the loss decreasing, as the amount of epochs increase. The loss and accuracy of the validation set plateaus after around epoch 3.
This might be due to the fact that the model overfits to the training data over time, while not learning abstract enough features that generalise to the validation set. The features that are learned are specific to the traning data, and thus score high on them. But the features might be idiosyncratic to the training set, and not be a general feature for the entire spectrum of digits, e.g. digits in the validation set.
"""

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
(x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = keras.datasets.cifar10.load_data()

# %%
x_train_cifar = x_train_cifar / 255
x_test_cifar = x_test_cifar / 255

# %%
y_train_cifar = keras.utils.to_categorical(y_train_cifar, 10)
y_test_cifar = keras.utils.to_categorical(y_test_cifar, 10)

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
    x_train_cifar,
    y_train_cifar,
    batch_size=32,
    epochs=20,
    verbose=1,
    validation_data=(x_test_cifar, y_test_cifar),
    shuffle=True
)

# %% 
show_training_plots(cifar_cnn_history)
