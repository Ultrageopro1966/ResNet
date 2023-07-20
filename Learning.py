import tensorflow as tf
from keras.losses import categorical_crossentropy
from keras.metrics import CategoricalAccuracy, Mean
from keras.optimizers import Adam

# Import necessary variables and data for model training and testing
from DataPreparation import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    test_dataset,
    train_dataset,
)

# Import ResNet defined layers
from ResNet import ResNet

# Create an instance of the ResNet model
model: ResNet = ResNet()

# Define the optimizer for model training
optimizer: tf.keras.optimizers.Optimizer = Adam(LEARNING_RATE)

# Define the loss function for model training
loss_func = categorical_crossentropy

# Define metrics to track model performance during training and testing
train_loss: Mean = Mean()
train_accuracy: CategoricalAccuracy = CategoricalAccuracy()
test_accuracy: CategoricalAccuracy = CategoricalAccuracy()

# Define the training step function (train_step) using the @tf.function decorator
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        prediction: tf.Tensor = model(images, training=True)
        loss: tf.Tensor = loss_func(labels, prediction)
    gradients: tf.Tensor = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, prediction)

# Define the testing step function (test_step) using the @tf.function decorator
@tf.function
def test_step(images, labels):
    prediction: tf.Tensor = model(images)
    test_accuracy(labels, prediction)

# Import necessary libraries for plotting the accuracy graph during training
from matplotlib import pyplot as plt

# Create a list to store accuracy values during training
accuracy_y_data: list = []

# Loop through the specified number of epochs for model training
for epoch in range(EPOCHS):
    batch: int = 0
    for images, labels in train_dataset:
        train_step(images, labels)
        accuracy_y_data.append(float(train_accuracy.result().numpy()) * 100)
        batch += 1

        # Display the training progress at each step (epoch and batch)
        print(" " * 100, end="\r")
        print(
            f"EPOCH {epoch + 1}/{EPOCHS}\tBATCH {batch}/{tf.data.experimental.cardinality(train_dataset)} ({round(batch/int(tf.data.experimental.cardinality(train_dataset)) * 100)}%)\tloss = {round(float(train_loss.result().numpy()), 3)}\taccuracy = {round(train_accuracy.result().numpy() * 100)}%",
            end="\r",
        )

# Save the trained model weights
model.save_weights("ResNet_Weights.h5")

# Test the model on the test dataset
print("\nModel testing...")
for images, labels in test_dataset:
    test_step(images, labels)

# Display the accuracy on the test dataset
print(f"Test Accuracy {round(test_accuracy.result().numpy() * 100, 3)}%")

# Plot the model accuracy graph
plt.plot(
    accuracy_y_data,
    c="b",
    label=f"Accuracy during training ({round(max(accuracy_y_data), 4)}%)",
)
plt.plot(
    [0, len(accuracy_y_data)],
    [
        round(test_accuracy.result().numpy() * 100, 3),
        round(test_accuracy.result().numpy() * 100, 3),
    ],
    c="r",
    label=f"Accuracy on test dataset ({round(test_accuracy.result().numpy() * 100, 3)}%)",
)
plt.xlabel("Batch")
plt.ylabel("Accuracy, %")
plt.title("Accuracy")

plt.grid(which="major")
plt.grid(which="minor", linestyle=":")
plt.legend()
plt.savefig("AccuracyGraph.png")
plt.show()
