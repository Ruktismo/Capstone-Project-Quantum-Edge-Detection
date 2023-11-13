import json
import logging
import os
import os.path as fs
from datetime import datetime

import numpy as np
from PIL import Image
from IPython.display import clear_output
import matplotlib.pyplot as plt
from qiskit.algorithms.optimizers import COBYLA  # TODO install qiskit_algorithms and swap this out
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
# Unused It's for splitting data into train test pools
# so after training we can use the test set to eval on data it has not seen before
from sklearn.model_selection import train_test_split

from Quantum.QCNN_circuit import build_qcnn

log = logging.getLogger("Quantum_Edge_Detection")

objective_func_vals = []
# Function for the NeuralNetworkClassifier to use to plot its progress.
def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()


"""
Will open a data files in folder in a depth first search pattern.
Sub-folder names will be assumed as labels for the data.
Folder structure should be the following.
path\
    label-1\
        data-point-1
        data-point-2
        data-point-3
        ...
    label-2\
    label-3\
    ...
"""
def open_data_folder(path, grouping="ALL", is_training_data=False):
    groupingTypes = ["OvO", "OvR", "ALL"]
    if grouping not in groupingTypes:
        log.error(f"Invalid grouping type, expected OvO, OvR, or ALL got {grouping}")
        exit(-10)
    data = []
    labels = []
    # gather all data in three groups
    for p in os.walk(path):
        # if just files, no dirs
        if len(p[1]) == 0:
            for f in p[2]:
                # load in photo
                file_path = os.path.join(p[0], f)
                img = np.asarray(Image.open(file_path)).flatten()  # load in the image as an array and flatten to 1D
                data.append(img)
                label = fs.basename(p[0]).lower()  # the folder name of the file should be the label

                # Determine label based off of grouping type
                # gather labels in left and right labels
                if grouping in groupingTypes[0]:
                    if ("left" in label) or ("right" in label):
                        labels.append(label)
                    elif "straight" in label:
                        data = data[:-1] # straight is not part of this data set remove it
                # gather labels in straight and turn labels
                elif grouping in groupingTypes[1]:
                    if "straight" in label:
                        labels.append("straight")
                    elif ("left" in label) or ("right" in label):
                        labels.append("turn")
                # labels are just dir names
                elif grouping in groupingTypes[2]:
                    labels.append(label)
                else:
                    log.error("We should not be here. You may have changed the labels from just Left, Straight, Right.")
                    exit(-20)

    if is_training_data:
        return np.asarray(data), np.asarray(labels)
    else:
        return np.asarray(data)


"""
Train the qcnn
Can take in a json file if available as a starting point for the network
"""
def train_qcnn(training_folder, start_point_json=None, output_file_name=None):
    # if output name is not provided just name it the date-time stamp
    if output_file_name is None:
        output_file_name = f"qcnn_train_point_{datetime.now().isoformat()}"
    
    if start_point_json is not None:
        with open(start_point_json, "r") as f:
            initial_point = json.load(f)
    else:
        log.warning("""
                No start point file provided. Initializing model to random values.\n
                NOTE: Starting from random will take much longer to train.
            """)
        initial_point = None

    fm, an = build_qcnn(20)  # build the feature map and ansatz for VQC
    # make Variational Quantum Classifier to deal with multi-label classification
    vqc = VQC(
        feature_map=fm,
        ansatz=an,
        loss="cross_entropy",
        optimizer=COBYLA(maxiter=200),
        callback=callback_graph,
        initial_point=initial_point
    )

    train_images, train_labels = open_data_folder(training_folder, is_training_data=True)

    # the docs put the objective_func_vals array here but python needs it in a bigger scope to work.
    objective_func_vals.clear()  # just empty it here to get the same effect
    plt.rcParams["figure.figsize"] = (12, 6)
    vqc.fit(train_images, train_labels)

    # score classifier
    classifier_score = f"Accuracy from the train data : {np.round(100 * vqc.score(train_images, train_labels), 2)}%"
    log.info(classifier_score)
    
    # save model to file
    vqc.save(output_file_name)
    
    return vqc


"""
Predict images using qcnn
qcnn_classifier: should be fully trained model, but can put untrained model here for before after comparison
image: A numPy array of image data, or a path to a file/directory containing image(s).
"""
def qcnn_predict(qcnn_classifier, image):
    # Check if image is a file or dir and open data accordingly.
    if isinstance(image, str):
        if fs.isfile(image):
            data = np.asarray(Image.open(image))
        elif fs.isdir(image):
            data = open_data_folder(image)
    # if it's not a path then it must be an np.array

    predictions = qcnn_classifier.predict(data)
    return predictions


if __name__ == "__main__":
    # train model on cropped images, un-cropped would be way too big for one QCNN
    train_qcnn("../Post-Proccessed_Images/400x160-cropped")
