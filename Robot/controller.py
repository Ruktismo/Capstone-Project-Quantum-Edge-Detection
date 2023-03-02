import time
from smbus2 import SMBus

# region # =========================== Initialize =========================== #
# ----- addressing the bus
I2C_BUS = SMBus(1)
I2C_ADDRESS = 0x18
I2C_COMMAND = 0xff

# ----- Motor Controls
I2C_STOP = 0x210A  # ...................... All motors stop
I2C_FORWARD = 0x220A  # ................... Left and right forward
I2C_BACKWARD = 0x230A  # .................. Left and right backwards
I2C_LEFT = 0x240A  # ...................... Left forward and right backwards
I2C_RIGHT = 0x250A  # ..................... Right forward and left backwards

I2C_LEFT_SLOW = 0x2605  # ................. Min left side speed
I2C_LEFT_FAST = 0x260A  # ................. Max left side speed
I2C_RIGHT_SLOW = 0x2705  # ................ Min right side speed
I2C_RIGHT_FAST = 0x270A  # ................ Max right side speed

I2C_L_HEADLIGHT_OFF = 0x3600  # ........... Left headlight off
I2C_L_HEADLIGHT_ON = 0x3601  # ............ Left headlight on
I2C_R_HEADLIGHT_OFF = 0x3700  # ........... Right headlight off
I2C_R_HEADLIGHT_ON = 0x3701  # ............ Right headlight on


# endregion


# region # =========================== State Functions =========================== #
def setstate_fast():
    # ----- Initialize speed to fast for both motors
    I2C_BUS.write_word_data(I2C_ADDRESS, I2C_COMMAND, I2C_LEFT_FAST)  # left motor set speed
    time.sleep(0.01)
    I2C_BUS.write_word_data(I2C_ADDRESS, I2C_COMMAND, I2C_RIGHT_FAST)  # right motor set speed
    time.sleep(0.01)


def setstate_slow():
    # ----- Initialize speed to slow for both motors
    I2C_BUS.write_word_data(I2C_ADDRESS, I2C_COMMAND, I2C_LEFT_SLOW)  # change left speed to slow
    time.sleep(0.01)
    I2C_BUS.write_word_data(I2C_ADDRESS, I2C_COMMAND, I2C_RIGHT_SLOW)  # change right speed to slow
    time.sleep(0.01)


# endregion


# region # =========================== Testing Functions =========================== #
def test_movement():
    # ----- Move car forward then backwards
    setstate_fast()  # set speed to fast
    I2C_BUS.write_word_data(I2C_ADDRESS, I2C_COMMAND, I2C_FORWARD)  # move forward for 0.5 seconds
    time.sleep(0.5)
    I2C_BUS.write_word_data(I2C_ADDRESS, I2C_COMMAND, I2C_STOP)  # stop
    time.sleep(0.01)
    I2C_BUS.write_word_data(I2C_ADDRESS, I2C_COMMAND, I2C_BACKWARD)  # move backwards for 0.5 seconds
    time.sleep(0.5)
    I2C_BUS.write_word_data(I2C_ADDRESS, I2C_COMMAND, I2C_STOP)  # stop
    time.sleep(0.01)

    # ----- Slowly turn in a circle - left then right
    setstate_slow()  # set speed to slow
    I2C_BUS.write_word_data(I2C_ADDRESS, I2C_COMMAND, I2C_LEFT)  # left turn for 2 seconds
    time.sleep(2)
    I2C_BUS.write_word_data(I2C_ADDRESS, I2C_COMMAND, I2C_RIGHT)  # right turn for 2 seconds
    time.sleep(2)
    I2C_BUS.write_word_data(I2C_ADDRESS, I2C_COMMAND, I2C_STOP)  # stop
    time.sleep(0.01)


def test_headlights():
    # ----- flash headlights
    for i in range(5):
        I2C_BUS.write_word_data(I2C_ADDRESS, I2C_COMMAND, I2C_L_HEADLIGHT_ON)
        time.sleep(0.01)
        I2C_BUS.write_word_data(I2C_ADDRESS, I2C_COMMAND, I2C_R_HEADLIGHT_ON)
        time.sleep(0.25)
        I2C_BUS.write_word_data(I2C_ADDRESS, I2C_COMMAND, I2C_L_HEADLIGHT_OFF)
        time.sleep(0.01)
        I2C_BUS.write_word_data(I2C_ADDRESS, I2C_COMMAND, I2C_R_HEADLIGHT_OFF)
        time.sleep(0.25)


# endregion


# region # =========================== Run Functions =========================== #
def run_forward(run_time, speed):
    # ----- set speed
    if speed == 1:
        setstate_fast()
    if speed == 0:
        setstate_slow()

    # ----- movement
    I2C_BUS.write_word_data(I2C_ADDRESS, I2C_COMMAND, I2C_FORWARD)  # move forward for given time
    time.sleep(run_time)
    I2C_BUS.write_word_data(I2C_ADDRESS, I2C_COMMAND, I2C_STOP)  # stop
    time.sleep(0.01)


def run_backwards(run_time, speed):
    # ----- set speed
    if speed == 1:
        setstate_fast()
    if speed == 0:
        setstate_slow()

    # ----- movement
    I2C_BUS.write_word_data(I2C_ADDRESS, I2C_COMMAND, I2C_BACKWARD)  # move forward for given time
    time.sleep(run_time)
    I2C_BUS.write_word_data(I2C_ADDRESS, I2C_COMMAND, I2C_STOP)  # stop
    time.sleep(0.01)


def run_left_turn(run_time, speed):
    # ----- set speed
    if speed == 1:
        setstate_fast()
    if speed == 0:
        setstate_slow()

    # ----- movement
    I2C_BUS.write_word_data(I2C_ADDRESS, I2C_COMMAND, I2C_LEFT)  # left turn for given time
    time.sleep(run_time)
    I2C_BUS.write_word_data(I2C_ADDRESS, I2C_COMMAND, I2C_STOP)  # stop
    time.sleep(0.01)


def run_right_turn(run_time, speed):
    # ----- set speed
    if speed == 1:
        setstate_fast()
    if speed == 0:
        setstate_slow()

    # ----- movement
    I2C_BUS.write_word_data(I2C_ADDRESS, I2C_COMMAND, I2C_RIGHT)  # right turn for given time
    time.sleep(run_time)
    I2C_BUS.write_word_data(I2C_ADDRESS, I2C_COMMAND, I2C_STOP)  # stop
    time.sleep(0.01)


# endregion


# region # =========================== Default run: testing =========================== #
def main():
    # ----- call pre-made test functions
    test_movement()
    test_headlights()


if __name__ == "__main__":
    main()

# endregion
