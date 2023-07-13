"""
Main Controller

This file coordinates all components of the project and handles timeing each section.
"""
# Standard modules used
import logging
import time
# Our modules used
import Quantum.QED as QED

# set up logger file and formatting.
logging.basicConfig(filename="Latest_Run.log", filemode='w', encoding='utf-8', level=logging.DEBUG,
                    format="%(asctime)s : %(levelname)s : %(name)s : %(funcName)s : %(message)s",
                    datefmt='%m/%d %I:%M:%S %p')
log = logging.getLogger(__name__)  # get logger obj, using __name__ for simplicity.

"""
Header function bellow are for any pre/post processing that needs to be done to keep things organized.
Right now they just have some logging and timing, fill in all of your stuff inbetween.
"""
# TODO fill in header functions
def connect_car():
    log.info("Connecting to car...")

    log.info("Connected to car, starting drive.")


def get_photo():
    log.debug("Taking photo")
    start_time = time.perf_counter()

    step_1 = time.perf_counter() - start_time
    log.debug(f"Got photo in {step_1:0.4f}sec")

def get_edges(pic):
    log.debug("Getting edges of photo")
    start_time = time.perf_counter()
    QED.QED(pic)
    step_2 = time.perf_counter() - start_time
    log.debug(f"Got edges of photo in {step_2:0.4f}sec")


def decide_drive_command():
    log.debug("Sending to Neural Network to decide movement")
    start_time = time.perf_counter()

    step_3 = time.perf_counter() - start_time
    log.debug(f"Movement decided in {step_3:0.4f}sec")


def send_command():
    log.debug("Sending movement command to car")
    start_time = time.perf_counter()

    step_4 = time.perf_counter() - start_time
    log.debug(f"Movement completed in {step_4:0.4f}sec")


def main():
    # TODO do we need this?
    log.info("Capstone Project 2023- Quantum Vision Robot Car\nBy:"
             "\n\tAndrew Erickson"
             "\n\tYumi Lamansky"
             "\n\tMeredith Kuhler"
             "\n\tMichael Del Regno"
             "\n\tKenneth Wang"
             "\n\tZaid Buni")
    connect_car()  # Call car to initiate startup.

    """
    This will be the format if we do a segmented approach. ie take photo -> process -> move -> loop.
    Its simple but lots of parts may be waiting when they could be doing somthing.
    We should do this approach first to get a sense of how long each step will take.
    
    If we want to make more of a pipeline so all 4 parts working all the time,
    then each step will need its own process.
    """
    # Loop until Ctrl+C (or maybe Neural Network can give stop)
    try:
        while True:
            cycle_start = time.perf_counter()
            # 1) Get photo from car
            get_photo()
            # 2) Pass photo to QED to get edge photo
            get_edges()
            # 3) Pass edge photo to Neural Network to get drive command
            decide_drive_command()
            # 4) Send drive command to car
            send_command()
            cycle_end = time.perf_counter()
            log.info(f"Movement cycle done in {(cycle_end - cycle_start):0.4f} seconds")
    except KeyboardInterrupt:
        log.info("Caught KeyboardInterrupt. Stopping")

    # disconnect with the car? (IDK if that has to be done)


if __name__ == "__main__":
    main()
