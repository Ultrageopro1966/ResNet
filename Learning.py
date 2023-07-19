import tensorflow as tf
from keras.losses import categorical_crossentropy
from keras.metrics import CategoricalAccuracy, Mean
from keras.optimizers import Adam

# Импорт необходимых переменных и данных для обучения и тестирования модели
from DataPreparation import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    test_dataset,
    train_dataset,
)

# Импорт определенных слоев ResNet
from ResNet import ResNet

# Создание экземпляра модели ResNet
model: ResNet = ResNet()

# Определение оптимизатора для обучения модели
optimizer: tf.keras.optimizers.Optimizer = Adam(LEARNING_RATE)

# Определение функции потерь (loss function) для обучения модели
loss_func = categorical_crossentropy

# Определение метрик для отслеживания производительности модели во время обучения и тестирования
train_loss: Mean = Mean()
train_accuracy: CategoricalAccuracy = CategoricalAccuracy()
test_accuracy: CategoricalAccuracy = CategoricalAccuracy()


# Определение функции обучающего шага (train_step) с использованием декоратора @tf.function
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        prediction: tf.Tensor = model(images, training=True)
        loss: tf.Tensor = loss_func(labels, prediction)
    gradients: tf.Tensor = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, prediction)


# Определение функции тестового шага (test_step) с использованием декоратора @tf.function
@tf.function
def test_step(images, labels):
    prediction: tf.Tensor = model(images)
    test_accuracy(labels, prediction)


# Импорт необходимых библиотек для построения графика точности в процессе обучения
from matplotlib import pyplot as plt

# Создание списка для сохранения значений точности в процессе обучения
accuracy_y_data: list = []

# Цикл обучения модели на заданное количество эпох
for epoch in range(EPOCHS):
    batch: int = 0
    for images, labels in train_dataset:
        train_step(images, labels)
        accuracy_y_data.append(float(train_accuracy.result().numpy()) * 100)
        batch += 1

        # Вывод прогресса обучения на каждом шаге (эпохе и пакете)
        print(" " * 100, end="\r")
        print(
            f"EPOCH {epoch + 1}/{EPOCHS}\tBATCH {batch}/{tf.data.experimental.cardinality(train_dataset)} ({round(batch/int(tf.data.experimental.cardinality(train_dataset)) * 100)}%)\tloss = {round(float(train_loss.result().numpy()), 3)}\taccuracy = {round(train_accuracy.result().numpy() * 100)}%",
            end="\r",
        )

# Сохранение обученной модели
model.save_weights("ResNet_Weights.h5")

# Тестирование модели на тестовом наборе данных
print("\nModel testing...")
for images, labels in test_dataset:
    test_step(images, labels)

# Вывод точности на тестовом наборе данных
print(f"Test Accuracy {round(test_accuracy.result().numpy() * 100, 3)}%")

# Построение графика точности модели
plt.plot(
    accuracy_y_data,
    c="b",
    label=f"Accuracy в процессе обучения ({round(max(accuracy_y_data), 4)}%)",
)
plt.plot(
    [0, len(accuracy_y_data)],
    [
        round(test_accuracy.result().numpy() * 100, 3),
        round(test_accuracy.result().numpy() * 100, 3),
    ],
    c="r",
    label=f"Accuracy на тестовой выборке ({round(test_accuracy.result().numpy() * 100, 3)}%)",
)
plt.xlabel("Batch")
plt.ylabel("Accuracy, %")
plt.title("Accuracy модели")

plt.grid(which="major")
plt.grid(which="minor", linestyle=":")
plt.legend()
plt.savefig("AccuracyGraph.png")
plt.show()
