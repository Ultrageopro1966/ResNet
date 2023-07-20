from configparser import (
    ConfigParser,  # Importing ConfigParser class to work with configuration files
)

import tensorflow as tf  # Importing TensorFlow
from keras import layers
from keras.datasets import mnist  # Importing MNIST data from Keras
from keras.utils import (
    to_categorical,  # Importing to_categorical function for one-hot encoding of labels
)

# Creating an instance of ConfigParser to read parameters from the configuration file
parser: ConfigParser = ConfigParser()
parser.read("configs.ini")

# Getting the value of learning_rate from the configuration and assigning it to the variable LEARNING_RATE
LEARNING_RATE: float = float(parser["DEFAULT"]["learning_rate"])

# Getting the value of batch_size from the configuration and assigning it to the variable BATCH_SIZE
BATCH_SIZE: int = int(parser["DEFAULT"]["batch_size"])

# Getting the value of epochs from the configuration and assigning it to the variable EPOCHS
EPOCHS: int = int(parser["DEFAULT"]["epochs"])

# Loading MNIST data and splitting it into training and testing sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Converting labels to categorical (one-hot encoding)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Converting data to TensorFlow tensors
x_train, y_train = tf.convert_to_tensor(x_train), tf.convert_to_tensor(y_train)
x_test, y_test = tf.convert_to_tensor(x_test), tf.convert_to_tensor(y_test)

# Normalizing data (scaling pixel values to the range [0, 1])
x_train /= 255
x_test /= 255

# Creating a training dataset as a TensorFlow Dataset
train_dataset: tf.data.Dataset = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(1000)  # Shuffling data
    .batch(BATCH_SIZE)  # Splitting into batches
)

# Creating a test dataset as a TensorFlow Dataset
test_dataset: tf.data.Dataset = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .shuffle(1000)  # Shuffling data
    .batch(BATCH_SIZE)  # Splitting into batches
)
