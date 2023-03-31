import sys
import time
from smbus2 import SMBus
from threading import Lock, Thread, Event
from typing import Tuple
import numpy as np
import logging


# ================================= Buffer Class ================================= #
class Buffer:
    def __init__(self, initial_value=None) -> None:
        self.buffer = initial_value
        self.lock = Lock()
        self.publish_event = Event()

    # ----- Read the current value of the buffer
    def read(self, timeout=None):
        current_value = None

        timed_out = not self.publish_event.wait(timeout)
        if timed_out:
            raise TimeoutError("A value was not published to the buffer within the timeout.")

        with self.lock:
            current_value = np.copy(self.buffer)
        self.publish_event.clear()
        return current_value

    # ----- Update the value of the buffer
    def write(self, value):
        with self.lock:
            self.buffer = np.copy(value)
        self.publish_event.set()

    # ----- Read from the buffer and write to it in a single critical section
    def read_and_write(self, value):
        current_value = None
        with self.lock:
            current_value = np.copy(self.buffer)
            self.buffer = value
        self.publish_event.set()
        return current_value


# ================================= Motor Constants ================================= #
# Addressing the I2C bus
I2C_ADDRESS = 0x18  # ........................ I2C bus address
I2C_COMMAND = 0xff  # ........................ I2C command address

# declaring speeds
I2C_LEFT_SPEED_SLOW = 0x2605  # .............. min left side speed
I2C_LEFT_SPEED_FAST = 0x260A  # .............. max left side speed
I2C_RIGHT_SPEED_SLOW = 0x2705  # ............. min right side speed
I2C_RIGHT_SPEED_FAST = 0x270A  # ............. max right side speed

# motor direction commands
I2C_STOP = 0x210A  # ......................... all motors stop
I2C_FORWARD = 0x220A  # ...................... forward
I2C_BACKWARD = 0x230A  # ..................... backward
I2C_LEFT = 0x240A  # ......................... left turn
I2C_RIGHT = 0x250A  # ........................ right turn

# Headlight controls
I2C_HEADLIGHT_LEFT_OFF = 0x3600  # ........... left headlight off
I2C_HEADLIGHT_LEFT_ON = 0x3601  # ............ left headlight on
I2C_HEADLIGHT_RIGHT_OFF = 0x3700  # .......... right headlight off
I2C_HEADLIGHT_RIGHT_ON = 0x3701  # ........... right headlight on

# camera servo control
I2C_SERVO_RANGE = [0x0000, 0x00FF]  # ........ camera range of motion


# ================================= Motor Operation ================================= #
class Motors:
    # ----- variable declaration
    motor_hex_base = 0x2600  # base value of hex I2C commands
    motors = [motor_hex_base, motor_hex_base + 0x100]
    min_speed = 0x1

    #
    # ----- constructor
    def __init__(self, ports=(3, 5)) -> None:
        # initialize
        self.i2c_bus_lock = Lock()
        self.bus = None
        self.servo_position = -1
        self.stop = False

        # create buffer
        self.motor_dict_buffer = Buffer({
            "servos": [
                {"isTurn": False, "servoId": ports[0], "servoSpeed": 0},
                {"isTurn": False, "servoId": ports[1], "servoSpeed": 0}
            ]
        })
        self.motor_speed_buffer = Buffer((None, None))
        self.last_target_speeds = None

        # create thread
        self.t = Thread(self.update_motor_speeds, ())
        self.t.start()

    #
    # ----- enter motor context
    def __enter__(self):
        return self

    #
    # ----- exit motor context
    def __exit__(self, type, value, traceback):
        # Stop the motors when the program exits so the robot doesn't fly away
        self.set_target_speed((0, 0))

        self.stop = True
        if self.t is not None:
            logging.debug("Waiting for Motors worker thread to finish...")
            self.t.join()
        logging.debug("Motors is exiting.")

    #
    # ----- send command to the bus for execution
    def send_command(self, word) -> None:
        with self.i2c_bus_lock:
            if self.bus is not None:
                self.bus.write_word_data(I2C_ADDRESS, I2C_COMMAND, word)
                time.sleep(0.05)  # min wait time to avoid overlapping commands

    #
    # ----- control motor speed
    def update_motor_speeds(self) -> None:
        with self.i2c_bus_lock:
            logging.debug("Starting I2C Bus...")
            self.bus = SMBus(1)
            time.sleep(1)  # wait for I2C bus initialization
            logging.debug("I2C Bus initialized.")

        while not self.stop:  # read current speed
            new_speeds, new_powers = (None, None)
            try:
                new_speeds, new_powers = self.motor_speed_buffer.read(0.5)
            except TimeoutError:
                continue

            if new_powers is not None:
                logging.debug(f"Setting motors speeds to new values: {new_speeds}")
                for power in new_powers:
                    self.send_command(int(power))
                    time.sleep(0.05)  # min wait for commands not to overlap
                logging.debug(f"Finished setting motor speeds: {new_speeds}")

                # start movement
                l, r = new_speeds
                command = None
                if l > 0 and r > 0:
                    command = I2C_FORWARD
                elif l < 0 and r < 0:
                    command = I2C_BACKWARD
                elif l == 0 and r == 0:
                    command = I2C_STOP
                elif l <= 0 <= r:
                    command = I2C_LEFT
                elif l >= 0 >= r:
                    command = I2C_RIGHT

                # send command
                logging.debug(f"Setting motor motion for: {new_speeds}")
                self.send_command(command)
                # time.sleep(0.01)

    #
    # ----- convert power
    def convert_speeds_to_commands(self, speeds: Tuple[np.double, np.double]) -> Tuple[int, int]:
        powers = []
        for idx, speed in enumerate(speeds):  # Get speed magnitude and convert from range [-1, 1] to [0, 10]
            offset = abs(int(speed * 10))
            motor = self.motors[idx]
            power = motor + self.min_speed + offset
            powers.append(int(power))
        return tuple(powers)

    #
    # ----- motor target speeds
    def set_target_speed(self, new_speeds: Tuple[np.double, np.double]) -> None:
        if self.last_target_speeds != new_speeds:
            powers = self.convert_speeds_to_commands(new_speeds)
            self.last_target_speeds = new_speeds
            logging.debug(f"Setting motor target speeds: {new_speeds}")
            self.motor_speed_buffer.write((new_speeds, powers))

    #
    # ----- headlight control
    def headlights(self, state=True):
        commands = (I2C_HEADLIGHT_LEFT_ON, I2C_HEADLIGHT_RIGHT_ON)
        if not state:
            commands = (I2C_HEADLIGHT_LEFT_OFF, I2C_HEADLIGHT_RIGHT_OFF)

        for command in commands:
            self.send_command(command)
            time.sleep(0.5)

    #
    # ----- camera servo control
    def set_servo_position(self, position):
        # only servo 8 is connected
        servo = 8
        servo_offset = 0x100 * servo
        if self.servo_position != position:
            self.servo_position = position
            command = servo_offset + np.clip(position, I2C_SERVO_RANGE[0], I2C_SERVO_RANGE[1])
            self.send_command(command)

    #
    # ----- get servo position
    def get_servo_position(self):
        return self.servo_position


# ================================= main function ================================= #
def main():
    motors = Motors()

    # ----- main loop
    while True:
        # read input
        command = input("> ")
        args = command.split(" ")

        if len(sys.argv) > 1:  # drive time variable
            drive_time = args[1]
        else:
            drive_time = 0.28  # default timing for drive straight

        #
        # ----- non-driving actions
        if args[0] == "servo":
            motors.set_servo_position(int(args[1]))
            print(motors.get_servo_position())
        elif args[0] == "lights":
            motors.headlights(True if args[1] == "True" else False)

        #
        # ----- driving actions
        elif args[0] == "f":  # ..................... forward
            motors.send_command(I2C_FORWARD)
            time.sleep(float(drive_time))
            motors.send_command(I2C_STOP)
        elif args[0] == "b":  # ..................... backwards
            motors.send_command(I2C_BACKWARD)
            time.sleep(float(drive_time))
            motors.send_command(I2C_STOP)
        elif args[0] == "l":  # ..................... left turn
            motors.send_command(I2C_LEFT)
            time.sleep(float(drive_time))
            motors.send_command(I2C_STOP)
        elif args[0] == "r":  # ..................... right turn
            motors.send_command(I2C_RIGHT)
            time.sleep(float(drive_time))
            motors.send_command(I2C_STOP)
        elif args[0] == "s":  # ..................... stop
            motors.send_command(I2C_STOP)
        elif args[0] == "q":  # ..................... exit the program
            sys.exit()


if __name__ == "__main__":
    main()
