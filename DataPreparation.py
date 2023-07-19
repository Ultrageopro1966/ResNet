from configparser import (
    ConfigParser,  # Импорт класса ConfigParser для работы с файлами конфигурации
)

import tensorflow as tf  # Импорт TensorFlow
from keras.datasets import mnist  # Импорт данных MNIST из Keras
from keras.utils import (
    to_categorical,  # Импорт функции to_categorical для преобразования меток в one-hot encoding
)

# Создание экземпляра ConfigParser для чтения параметров из файла конфигурации
parser: ConfigParser = ConfigParser()
parser.read("configs.ini")

# Получение значения learning_rate из конфигурации и присвоение его переменной LEARNING_RATE
LEARNING_RATE: float = float(parser["DEFAULT"]["learning_rate"])

# Получение значения batch_size из конфигурации и присвоение его переменной BATCH_SIZE
BATCH_SIZE: int = int(parser["DEFAULT"]["batch_size"])

# Получение значения epochs из конфигурации и присвоение его переменной EPOCHS
EPOCHS: int = int(parser["DEFAULT"]["epochs"])

# Загрузка данных MNIST и разделение на тренировочный и тестовый наборы
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Преобразование меток в категориальные (one-hot encoding)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Преобразование данных в тензоры TensorFlow
x_train, y_train = tf.convert_to_tensor(x_train), tf.convert_to_tensor(y_train)
x_test, y_test = tf.convert_to_tensor(x_test), tf.convert_to_tensor(y_test)

# Нормализация данных (приведение значений пикселей к диапазону [0, 1])
x_train /= 255
x_test /= 255

# Создание тренировочного набора данных в виде TensorFlow Dataset
train_dataset: tf.data.Dataset = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(1000)  # Перемешивание данных
    .batch(BATCH_SIZE)  # Разбиение на батчи
)

# Создание тестового набора данных в виде TensorFlow Dataset
test_dataset: tf.data.Dataset = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .shuffle(1000)  # Перемешивание данных
    .batch(BATCH_SIZE)  # Разбиение на батчи
)
