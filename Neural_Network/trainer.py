import tensorflow as tf
from tensorflow import keras
from tensorflow.lite.python.interpreter import Interpreter
from keras.models import Sequential
from keras import layers
import keras

import sys
import os
import time
import logging
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# get logger
log = logging.getLogger("Quantum_Edge_Detection")
# TODO add arguments parser
modelFilePath = os.path.dirname(__file__)+"\\model.tflite"

class NeuralNetwork:
    def __init__(self, file=modelFilePath):
        self.file = file
        self.model = Interpreter(model_path=self.file)
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

    def reload_model(self):
        self.model = Interpreter(model_path=self.file)
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

    # prediction on a full 640x480 images takes around 0.03 sec, on a laptop
    def predict(self, pic):
        P = pic[np.newaxis, :, :, np.newaxis]  # model is expecting a shape of (0, imgX, imgY, 0)
        # set input for prediction
        self.model.set_tensor(self.input_details[0]["index"], P)
        # preform prediction
        self.model.invoke()
        # get prediction
        prediction = self.model.get_tensor(self.output_details[0]["index"])
        # prediction is an array of probabilities (0.0 to 1.0) of the format ["Right", "Straight", "Left"]
        log.debug(f"Prediction [r,s,l]: {prediction}")
        prediction = prediction.tolist()[0]
        i = prediction.index(max(prediction))
        if prediction[i] < 0.5:
            log.warning("Model seems uncertain on this image")
            exit(-20)
        if i == 0:
            return 'r'
        elif i == 1:
            return 's'
        elif i == 2:
            return 'l'
        else:
            log.error("We should never be here. (NN couldn't make a decision)")
            exit(-10)

class NeuralNetworkTrainer:
    def __init__(self, data_path="../Post-Proccessed_images", model_name="model.tflite"):
        self.epochs = 10
        self.trainedModel = None
        self.history = None
        self.data_path = data_path
        self.model_name = model_name

    def train_model(self):
        data_dir = Path(self.data_path)
        img_width = 640  # 640
        img_height = 480  # 480
        batch_size = 32
        class_names = ["Right", "Straight", "Left"]
        log.debug(f"Loading in dataset from {self.data_path}")
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
            #print(image_batch.shape)
            #print(labels_batch)
            break
        # TODO IDK if we should log these print statements
        #print(train_ds, val_ds)
        #print(np.shape(train_ds), np.shape(val_ds))
        log.debug("Dataset loaded in, building model")
        AUTOTUNE = tf.data.AUTOTUNE

        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

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
        log.debug("Model built, Starting Training")
        history = model.fit(train_ds, validation_data=val_ds, epochs=self.epochs)
        log.debug("Training complete")
        self.trainedModel = model
        self.history = history

    def visualize(self):
        acc1 = self.history.history["accuracy"]
        val_acc1 = self.history.history["val_accuracy"]
        acc = self.history.history["sparse_top_k_categorical_accuracy"]
        val_acc = self.history.history["val_sparse_top_k_categorical_accuracy"]

        loss = self.history.history["loss"]
        val_loss = self.history.history["val_loss"]

        epochs_range = range(self.epochs)

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

    def export_model(self):
        if self.trainedModel is None:
            log.error("No model to save")
            return
        log.debug("Exporting the model to tflite file")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.trainedModel)
        tflite_model = converter.convert()

        log.debug(f"Saving as {self.model_name}")
        # Save the model.
        with open(self.model_name, "wb") as f:
            f.write(tflite_model)

        log.debug("Finished exporting the model.")

    def train(self, visualize=False):
        self.train_model()
        self.export_model()
        if visualize:
            self.visualize()


# Tester function to see if NN is working, by loading in the model and running a bunch of predictions
def run_predictions():
    data_dir = Path("../Post-Proccessed_Images")
    img_width = 640  # 640
    img_height = 480  # 480
    class_names = ["Right", "Straight", "Left"]

    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",  # labels are from directory names
        label_mode="int",
        class_names=class_names,
        color_mode="grayscale",  # use only 1 channel
        image_size=(img_height, img_width),  # resize input images down to 160x64
    )

    M = NeuralNetwork(file="./NewDatasetModel.tflite")
    i = 0
    times = []
    for f in ds.file_paths:
        img = Image.open(f)
        image = np.asarray(img, dtype=np.float32)
        tic = time.perf_counter()
        ans = M.predict(image)
        tok = time.perf_counter()
        times.append(tok - tic)
        print(f"{f}: {ans}")
        i += 1
        if i > 100:
            print(sum(times)/len(times))
            exit()


if __name__ == "__main__":
    # if we are main we have to set up the stream handler
    formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(name)s : %(funcName)s : %(message)s")
    # Stream handler to output to stdout
    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setLevel(logging.INFO)  # handlers can set their logging independently or take the parent.
    log_stream_handler.setFormatter(formatter)
    # add handlers to log
    log.addHandler(log_stream_handler)

    #NNT = NeuralNetworkTrainer(model_name="NewDatasetModel.tflite")
    #NNT.train(True)
    run_predictions()
