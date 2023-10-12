"""
Main Controller

This file coordinates all components of the project and handles timeing each section.
"""
# Standard modules used
import logging
import time
# Our modules used
import Quantum.QED as QED
import Robot.middleware as Robot
from random import *

# set up logger file and formatting.
logging.basicConfig(filename="Latest_Run.log", filemode='w', encoding='utf-8', level=logging.DEBUG,
                    format="%(asctime)s : %(levelname)s : %(name)s : %(funcName)s : %(message)s",
                    datefmt='%m/%d %I:%M:%S %p')
log = logging.getLogger(__name__)  # get logger obj, using __name__ for simpl3icity.
robot = Robot.Connection()

"""
Header function bellow are for any pre/post processing that needs to be done to keep things organized.
Right now they just have some logging and timing, fill in all of your stuff inbetween.
"""
# TODO fill in header functions
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
    edge_img = QED.QED("mostRecentPhoto")
    step_2 = time.perf_counter() - start_time
    log.debug(f"Got edges of photo in {step_2:0.4f}sec")
    return edge_img


def decide_drive_command(edge_pic):
    log.debug("Sending to Neural Network to decide movement")
    start_time = time.perf_counter()
    command = choice(["l", "f", "b", "r"])
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
    # TODO do we need this?
    log.info("Capstone Project 2023- Quantum Vision Robot Car\nBy: Andrew Erickson, Yumi Lamansky, Meredith Kuhler"
             ", Michael Del Regno, Kenneth Wang, Zaid Buni")
    connect_car()  # Call car to initiate startup.

    """
    This will be the format if we do a segmented approach. ie take photo -> process -> move -> loop.
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
    main()
