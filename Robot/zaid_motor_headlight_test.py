import time
from smbus2 import SMBus
from threading import Lock, Thread, Event
from typing import Tuple
import numpy as np
import logging

"""
Zaid: For purposes of testing and because imports were not working correctly
I manually added the Buffer class from common, the Motors class from motors
and the test code from motor_test.py to test if things worked
results: 1. Able to turn on/off the headlights, 2. Able to operate servo
"""

"""
Buffer
"""


class Buffer:
    def __init__(self, initial_value=None) -> None:
        self.buffer = initial_value
        self.lock = Lock()
        self.publish_event = Event()

    # Read the current value of the buffer
    def read(self, timeout=None):
        current_value = None

        timed_out = not self.publish_event.wait(timeout)
        if timed_out:
            raise TimeoutError("A value was not published to the buffer within the timeout.")

        with self.lock:
            current_value = np.copy(self.buffer)
        self.publish_event.clear()
        return current_value

    # Update the value of the buffer
    def write(self, value):
        with self.lock:
            self.buffer = np.copy(value)
        self.publish_event.set()

    # Read from the buffer and write to it in a single critical section
    def read_and_write(self, value):
        current_value = None
        with self.lock:
            current_value = np.copy(self.buffer)
            self.buffer = value
        self.publish_event.set()
        return current_value


"""
Motors
This class controls the motors, and more generally anything connected to the LABISTS
daughter board that uses the I2C bus, including the headlights and servo.

We are restricted in what we can do by the protocol of the I2C commands.
It is important that we don't send commands too fast, or else they will get eaten up by the I2C bus.
However, we want to send commands to the middleware really fast. The solution is to store the target
speed values in a buffer. We can send commands to update the target speeds instantaneously.
Another thread continuously sends the commands to the I2C bus only when the target speed changes,
or as fast as it is able to.

We're also limited by the physical capabilities of the motors. The motors are geared to have a high speed
but low torque. This means that at the lowest speed setting, it does not take much resistance for the motors
to stall and fail to spin. However, once you set the speed high enough to overcome the static friction, 
the gearing is too high, and the robot moves too fast to be able to reliably process the camera feed in time.
The solution is to implement some sort of PWM control in VIPLE. Turn the motors on at a some power for a short
period of time, and stop for some period of time. By varying duty cycle, you can control the speed in a more
fine-grained manner, at the cost of a more jerky motion. We're still limited by the speed at which we
can send commands down the I2C bus; if we send commands at too high of a frequency, the robot will skip
phases in the PWM cycle, which manifests as an uneven, jerking motion.
"""

# Constants for addressing the I2C bus
I2C_ADDRESS = 0x18
I2C_COMMAND = 0xff

# Motor command constants
# Send these commands on the I2C to control the motors/lights

# Step 1) Set the speed of the left/right motors
# You can't control the four motors independently, only the left and right sides
I2C_LEFT_SPEED_SLOW = 0x2605  # minimum left side speed
I2C_LEFT_SPEED_FAST = 0x260A  # maximum left side speed
I2C_RIGHT_SPEED_SLOW = 0x2705  # minimum right side speed
I2C_RIGHT_SPEED_FAST = 0x270A  # maximum right side speed

# Step 2) Control the direction of the motors
I2C_STOP = 0x210A  # all motors stop
I2C_FORWARD = 0x220A  # left and right move forward
I2C_BACKWARD = 0x230A  # left and right move backward
I2C_LEFT = 0x240A  # left moves backward, right moves forward
I2C_RIGHT = 0x250A  # right moves backward, left moves forward

# Headlight controls
I2C_HEADLIGHT_LEFT_OFF = 0x3600
I2C_HEADLIGHT_LEFT_ON = 0x3601
I2C_HEADLIGHT_RIGHT_OFF = 0x3700
I2C_HEADLIGHT_RIGHT_ON = 0x3701

# Servo controls
I2C_SERVO_RANGE = [0x0000, 0x00FF]


class Motors:
    motor_hex_base = 0x2600  # base value of hex I2C commands
    motors = [motor_hex_base, motor_hex_base + 0x100]
    min_speed = 0x1

    # Motors constructor
    def __init__(self, ports=(3, 5)) -> None:
        self.i2c_bus_lock = Lock()
        self.bus = None

        self.servo_position = -1
        self.stop = False

        self.motor_dict_buffer = Buffer({
            "servos": [
                {
                    "isTurn": False,
                    "servoId": ports[0],
                    "servoSpeed": 0
                },
                {
                    "isTurn": False,
                    "servoId": ports[1],
                    "servoSpeed": 0
                }]
        })

        self.motor_speed_buffer = Buffer((None, None))
        self.last_target_speeds = None

        self.t = Thread(target=self.update_motor_speeds, args=())
        self.t.start()

    # Entering the context of the motor object
    def __enter__(self):
        return self

    # When exiting the context of the motor, stop worker thread
    def __exit__(self, type, value, traceback):
        # Stop the motors when the program exits so the robot doesn't fly away
        self.set_target_speed((0, 0))

        self.stop = True
        if self.t is not None:
            logging.debug("Waiting for Motors worker thread to finish...")
            self.t.join()
        logging.debug("Motors is exiting.")

    # Send a command to the I2C bus
    def send_command(self, word) -> None:
        with self.i2c_bus_lock:
            if self.bus is not None:
                self.bus.write_word_data(I2C_ADDRESS, I2C_COMMAND, word)
                # IMPORTANT!!!!!!!!!
                # If you sent commands too quickly they will be lost, and your motors will not do what you tell them to!
                # Below is the minimum amount of time to wait but you should probably wait even longer than this
                time.sleep(0.05)

    # Constantly read the motor speed target, and send I2C commands to match the target
    def update_motor_speeds(self) -> None:
        with self.i2c_bus_lock:
            logging.debug("Starting I2C Bus...")
            self.bus = SMBus(1)
            time.sleep(1)  # You need to wait for the I2C bus to initialize
            logging.debug("I2C Bus initialized.")

        while not self.stop:
            # Read the current speed target from the buffer
            new_speeds, new_powers = (None, None)
            try:
                new_speeds, new_powers = self.motor_speed_buffer.read(
                    timeout=0.5)
            except TimeoutError:
                continue

            if new_powers is not None:
                logging.debug(
                    f"Setting motors speeds to new values: {new_speeds}")
                # If there was new_powers, send commands for L and R motors to match speed
                for power in new_powers:
                    self.send_command(int(power))
                    # You need to wait to make sure that the I2C bus doesn't eat up one of the commands!!!!!!!!
                    # time.sleep(0.05)
                logging.debug(f"Finished setting motor speeds: {new_speeds}")
                # Now that we've set the magnitude of the speed, we need to set the direction and start the motors moving
                l, r = new_speeds
                command = None
                if(l > 0 and r > 0):
                    command = I2C_FORWARD
                elif(l < 0 and r < 0):
                    command = I2C_BACKWARD
                elif(l == 0 and r == 0):
                    command = I2C_STOP
                elif(l <= 0 and r >= 0):
                    command = I2C_LEFT
                elif(l >= 0 and r <= 0):
                    command = I2C_RIGHT
                logging.debug(f"Setting motor motion for: {new_speeds}")
                self.send_command(command)
                # time.sleep(0.01)

    # Conver the power from double to the hex value expected by the I2C bus
    def convert_speeds_to_commands(self, speeds: Tuple[np.double, np.double]) -> Tuple[int, int]:
        powers = []
        for idx, speed in enumerate(speeds):
            # Get the magnitude of the speed and convert from range [-1, 1] to [0, 10]
            offset = abs(int(speed * 10))
            motor = self.motors[idx]
            power = motor + self.min_speed + offset
            powers.append(int(power))
        return tuple(powers)

    # Set the speed target for the motors
    def set_target_speed(self, new_speeds: Tuple[np.double, np.double]) -> None:
        if self.last_target_speeds != new_speeds:
            powers = self.convert_speeds_to_commands(new_speeds)
            self.last_target_speeds = new_speeds
            logging.debug(f"Setting motor target speeds: {new_speeds}")
            self.motor_speed_buffer.write((new_speeds, powers))

    # Set the state of the headlights
    def headlights(self, state=True):
        commands = (I2C_HEADLIGHT_LEFT_ON, I2C_HEADLIGHT_RIGHT_ON)
        if not state:
            commands = (I2C_HEADLIGHT_LEFT_OFF, I2C_HEADLIGHT_RIGHT_OFF)

        for command in commands:
            self.send_command(command)
            # For some reason, if you don't wait about half a second, usually only one headlight turns on
            time.sleep(0.5)

    # Set the servo position range [0,255]
    def set_servo_position(self, position):
        # only servo 8 is connected
        servo = 8
        servo_offset = 0x100 * servo
        if (self.servo_position != position):
            self.servo_position = position
            command = servo_offset + \
                np.clip(position, I2C_SERVO_RANGE[0], I2C_SERVO_RANGE[1])
            self.send_command(command)

    # Get the current position of the servo motor
    def get_servo_position(self):
        return self.servo_position



def main():
    motors = Motors()
    while True:
        command = input("> ")
        args = command.split(" ")
        if args[0] == "servo":
            motors.set_servo_position(int(args[1]))
            print(motors.get_servo_position())
        if args[0] == "headlights":
            motors.headlights(True if args[1] == "True" else False)
        if args[0] == "forward":
            motors.send_command(I2C_FORWARD)
            sleep_time = args[1]
            time.sleep(float(sleep_time))
            motors.send_command(I2C_STOP)
        if args[0] == "back":
            motors.send_command(I2C_BACKWARD)
            sleep_time = args[1]
            time.sleep(float(sleep_time))
            motors.send_command(I2C_STOP)
        if args[0] == "left":
            motors.send_command(I2C_LEFT)
            sleep_time = args[1]
            time.sleep(float(sleep_time))
            motors.send_command(I2C_STOP)
        if args[0] == "right":
            motors.send_command(I2C_RIGHT)
            sleep_time = args[1]
            time.sleep(float(sleep_time))
            motors.send_command(I2C_STOP)
        if args[0] == "stop":
            motors.send_command(I2C_STOP)



if __name__ == "__main__":
    main()
