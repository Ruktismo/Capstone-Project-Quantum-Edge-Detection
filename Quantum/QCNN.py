import json
import logging
import os
import os.path as fs
from datetime import datetime

import numpy as np
from PIL import Image
from IPython.display import clear_output
import matplotlib.pyplot as plt
from qiskit.algorithms.optimizers import COBYLA
#make sure to: pip install qiskit_algorithms
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
# Unused It's for splitting data into train test pools
# so after training we can use the test set to eval on data it has not seen before
from sklearn.model_selection import train_test_split

from Quantum.QCNN_circuit import build_qcnn

log = logging.getLogger(__name__)

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
        data-point-1.png
        data-point-2.png
        data-point-3.png
        ...
    label-2\
    label-3\
    ...
"""
# TODO test if this opens everything correctly
def open_data_folder(path, is_training_data=False):
    data = []
    labels = []
    for f in os.listdir(path):
        # if f is not one of the classification folders
        if fs.isdir(f):
            sub_folder_data = open_data_folder(os.path.join(path, f), is_training_data)
            data.append(sub_folder_data)
        else:
            file_path = os.path.join(path, f)
            img = np.asarray(Image.open(file_path))
            data.append(img)
            labels.append(fs.basename(f))

    if is_training_data:
        return data, labels
    else:
        return data


"""
Train the qcnn
Can take in a json file if available as a starting point for the network
"""
def train_qcnn(training_folder:str, start_point_json=None, output_file_name:str=None):
    # if output name is not provided just name it the date-time stamp
    if output_file_name is None:
        output_file_name = f"qcnn_train_point_{datetime.now().isoformat()}"
    
    if start_point_json is not None:
        with open(start_point_json, "r") as f:
            initial_point = json.load(f)
    else:
        log.warning("""
                No start point file provided. Initializing model to random values.\n
                NOTE: Starting from random will take a long time to train.
            """)
        initial_point = None
    
    classifier = NeuralNetworkClassifier(
                    build_qcnn(),
                    optimizer=COBYLA(maxiter=200),  # Set max iterations here
                    callback=callback_graph,
                    initial_point=initial_point,
                )

    train_images, train_labels = open_data_folder(training_folder, is_training_data=True)

    # the docs put the array here but python needs it in a bigger scope to work.
    objective_func_vals.clear()  # just empty it here to get the same effect
    plt.rcParams["figure.figsize"] = (12, 6)
    classifier.fit(train_images, train_labels)

    # score classifier
    classifier_score = f"Accuracy from the train data : {np.round(100 * classifier.score(train_images, train_labels), 2)}%"
    print(classifier_score)
    log.info(classifier_score)
    
    # save model to file
    classifier.save(output_file_name)
    
    return classifier


"""
Predict images using qcnn
qcnn_classifier: should be fully trained model, but can put untrained model here for before after comparison
image: A numPy array of image data, or a path to a file/directory containing image(s).
"""
def qcnn_predict(qcnn_classifier:NeuralNetworkClassifier , image:(np.array | str)):
    # Check if image is a file or dir and open data accordingly.
    if isinstance(image, str):
        if fs.isfile(image):
            data = np.asarray(Image.open(image))
        elif fs.isdir(image):
            data = open_data_folder(image)
    # if it's not a path then it must be an np.array

    predictions = qcnn_classifier.predict(data)
    return predictions