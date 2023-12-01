"""
Main Controller

This file coordinates all components of the project and handles timeing each section.
"""
# Standard modules used
import sys
import logging
import time
from random import *
# Our modules used
from Quantum.Quantum_Edge_Detection import QED
import Robot.middleware as Robot
from Neural_Network.trainer import NeuralNetwork

# set up logger file and formatting.
log = logging.getLogger("Quantum_Edge_Detection")  # get logger obj.
formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(name)s : %(funcName)s : %(message)s")
log.setLevel(logging.DEBUG)
# File handler for log to dump to
log_file_handler = logging.FileHandler("Latest_Run.log", mode='w')
log_file_handler.setFormatter(formatter)
# Stream handler to output to stdout
log_stream_handler = logging.StreamHandler(sys.stdout)
log_stream_handler.setLevel(logging.INFO)  # handlers can set their logging independently or take the parent.
log_stream_handler.setFormatter(formatter)
# add handlers to log
log.addHandler(log_file_handler)
log.addHandler(log_stream_handler)
"""
robot = Robot.Connection()
qed = QED()
NN = NeuralNetwork()"""

"""
Header functions bellow are for any pre/post processing that needs to be done to keep things organized.
Right now they just have some logging and timing, along with whatever call they need to make.
"""
def connect_car():
    log.info("Connecting to car...")
    robot.connect()
    log.info("Connected to car, starting drive.")


def get_photo():
    log.debug("Taking photo")
    start_time = time.perf_counter()
    robot.get_last_pic()
    step_1 = time.perf_counter() - start_time
    log.debug(f"Got photo in {step_1:0.4f}sec")
    return None


def get_edges(pic=None):
    log.debug("Getting edges of photo")
    start_time = time.perf_counter()
    edge_img = qed.run_QED("mostRecentPhoto.jpg")
    step_2 = time.perf_counter() - start_time
    log.debug(f"Got edges of photo in {step_2:0.4f}sec")
    return edge_img


def decide_drive_command(edge_pic):
    log.debug("Sending to Neural Network to decide movement")
    start_time = time.perf_counter()
    command = NN.predict(edge_pic)  # choice(["l", "f", "b", "r"])  # for random choice
    step_3 = time.perf_counter() - start_time
    log.debug(f"Movement decided in {step_3:0.4f}sec")
    return command


def send_command(command):
    log.debug("Sending movement command to car")
    start_time = time.perf_counter()
    robot.exec_control_command(command)
    step_4 = time.perf_counter() - start_time
    log.debug(f"Movement completed in {step_4:0.4f}sec")


def main():
    testing = False
    if testing:
        edge_pic = get_edges()
        command = decide_drive_command(edge_pic)
        print("NN decision: " + command)
        return



    # TODO do we need this?
    log.info("Capstone Project 2023- Quantum Vision Robot Car\nBy: Andrew Erickson, Yumi Lamansky, Meredith Kuhler"
             ", Michael Del Regno, Kenneth Wang, Zaid Buni")
    connect_car()  # Call car to initiate startup.

    """
    This will be the format for a segmented approach. ie take photo -> process -> move -> loop.
    Its simple but lots of parts may be waiting when they could be doing somthing.
    We should do this approach first to get a sense of how long each step will take.
    
    If we want to make more of a pipeline so all 4 parts working all the time,
    then each step will need its own process.
    """
    # Loop until Ctrl+C (or maybe Neural Network can give stop command?)
    try:
        while True:
            print("start cycle")
            cycle_start = time.perf_counter()
            # 1) Get photo from car
            get_photo()
            print("got photo")
            # 2) Pass photo to QED to get edge photo
            edge_pic = get_edges()
            print("got edges")
            # 3) Pass edge photo to Neural Network to get drive command
            command = decide_drive_command(edge_pic)
            # 4) Send drive command to car
            send_command(command)
            cycle_end = time.perf_counter()
            log.info(f"Movement cycle done in {(cycle_end - cycle_start):0.4f} seconds")
            print(f"Movement cycle done in {(cycle_end - cycle_start):0.4f} seconds")
    except KeyboardInterrupt:
        log.info("Caught KeyboardInterrupt. Stopping")

    # disconnect with the car? (IDK if that has to be done)


if __name__ == "__main__":
    robot = Robot.Connection()
    qed = QED()
    NN = NeuralNetwork()
    main()
