import tensorflow as tf
from tensorflow import keras
from tensorflow.lite.python.interpreter import Interpreter
from keras.models import Sequential
from keras import layers
import keras

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

epochs = 150

# TODO add arguments parser

class NerualNetwork:
    def __init__(self):
        self.model = Interpreter(model_path="./model.tflite")

        self.model.allocate_tensors()

        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

    def reload_model(self):
        self.model = Interpreter(model_path="./model.tflite")

        self.model.allocate_tensors()

        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

    def predict(self, pic):
        P = pic[np.newaxis, :, :, np.newaxis]  # model is expecting a shape of (0, imgX, imgY, 0)
        # set input for prediction
        self.model.set_tensor(self.input_details[0]["index"], P)
        # preform prediction
        self.model.invoke()
        # get prediction
        prediction = self.model.get_tensor(self.output_details[0]["index"])
        # prediction is an array of probabilities (0.0 to 1.0) of the format ["Right", "Straight", "Left"]
        print(prediction)
        i = prediction.index(max(prediction))
        if i == 0:
            return 'r'
        elif i == 1:
            return 's'
        elif i == 2:
            return 'l'


NN = NerualNetwork()

data_path = r"/Users/mgriffin/Documents/CapstoneRepo/Capstone-Project-AI-Robot-Car-Maze-Navigation/dataset_12_12_00_877387"


def train_model():
    data_dir = Path(data_path)
    img_width = 640  # 640
    img_height = 480  # 480
    batch_size = 32
    # class_names = ["LeftTurn", "OffsetLeft",
    # "Straight", "OffsetRight", "RightTurn"]
    class_names = ["Right", "Straight", "Left"]

    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",  # labels are from directory names
        label_mode="int",
        class_names=class_names,
        color_mode="grayscale",  # use only 1 channel
        seed=4,  # must set a specific seed for deterministic validation split
        image_size=(img_height, img_width),  # resize input images down to 160x64
        validation_split=0.2,  # use 20% of training images for validation
        subset="both",  # return training dataset and validation dataset
        batch_size=batch_size,
    )

    shape = None
    for image_batch, labels_batch in val_ds:
        shape = image_batch.shape
        print(image_batch.shape)
        print(labels_batch)
        break

    print(train_ds, val_ds)
    print(np.shape(train_ds), np.shape(val_ds))

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    augmentation_layer = keras.Sequential(
        [
            # layers.RandomFlip("horizontal", input_shape=(
            #    img_height, img_width, 3)),
            # layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    model = Sequential(
        [
            # augmentation_layer,
            layers.Rescaling(1.0 / 255, input_shape=(img_height, img_width, 1)),  # 3)),
            layers.SeparableConv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.SeparableConv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.SeparableConv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(rate=0.25),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(len(class_names), activation="softmax"),
        ]
    )

    model.build(shape)

    model.compile(
        optimizer="adam",
        # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        # metrics=["accuracy"],
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2)],
    )

    model.summary()

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    return model, history


def predict():
    model = Interpreter(model_path="./model.tflite")

    model.allocate_tensors()

    input_details = model.get_input_details()
    output_details = model.get_output_details()

    pic = "../Post-Proccessed_Images/Right/RIGHT_2022_11_28_23_45_15_picture_000000000 (59).jpg"
    P = np.asarray(Image.open(pic))
    P = P.astype(np.float32)            # convert array to float32 if it is not already
    P = P[np.newaxis, :, :, np.newaxis] # model is expecting a shape of (0, imgX, imgY, 0)
    # set input for prediction
    model.set_tensor(input_details[0]["index"], P)
    # preform prediction
    model.invoke()
    # get prediction
    prediction = model.get_tensor(output_details[0]["index"])
    # prediction is an array of probabilities (0.0 to 1.0) of the format ["Right", "Straight", "Left"]
    print(prediction)


def visualize(history):
    acc1 = history.history["accuracy"]
    val_acc1 = history.history["val_accuracy"]
    acc = history.history["sparse_top_k_categorical_accuracy"]
    val_acc = history.history["val_sparse_top_k_categorical_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.plot(epochs_range, acc1, label="Training Accuracy1")
    plt.plot(epochs_range, val_acc1, label="Validation Accuracy1")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.show()


def export_model(model):
    print("Exporting the model...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)

    print("Finished exporting the model.")


def main():
    (model, history) = train_model()
    export_model(model)
    visualize(history)


if __name__ == "__main__":
    main()
