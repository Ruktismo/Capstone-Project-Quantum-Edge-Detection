# Python script to execute commands over SSH connection
# requires paramiko
# FOR USE WITH CAPSTONE ROBOT: make sure to connect to robot's Wi-Fi before using.
# Wifi Information - Name: RPiRobot2, Pass:  password123
from pwn import *
# ==================== Server Credentials and Saved Globals ==================== #
robo_server = "192.168.16.2"
robo_port = 22
robo_user = "pi"
robo_password = "raspberry"
fps = 12

robo_control_path = "/home/pi/Desktop/newCapstone/middleware/devices/"
robo_control_base = "python3 controller.py"
robo_pwn_controller = "python3 controllerPwn.py"
robo_photo_path = "/tmp/pics"
robo_last_photo = "lastPic"
robo_cam_cmd = f"sudo LD_LIBRARY_PATH='pwd' ./mjpg_streamer -i \"input_uvc.so -d /dev/video0 -f {fps}\" -o \"./output_file.so -f {robo_photo_path} -l {robo_last_photo}\""

robo_forward = robo_control_base + " --command f "
robo_backwards = robo_control_base + " --command b "
robo_left = robo_control_base + " --command l "
robo_right = robo_control_base + " --command r "
robo_stop = robo_control_base + " --command s --runtime 0"
robo_servo = robo_control_base + " --command servo --runtime 0"
robo_lights = robo_control_base + " --command lights --runtime 0"


# ==================== SSH Control Commands ==================== #
class Connection:
    def __init__(self):
        self.shell = ssh(host='192.168.16.2', user='pi', password='raspberry')
        self.shell.run('mkdir /tmp/pics')
        self.shell.set_working_directory('/tmp/pics')
        self.controller = self.shell.process(['python3', robo_pwn_controller], cwd="/home/pi/Desktop/newCapstone/middleware/devices")

    def connect(self):
        # only reconnect if we lost it
        if not self.shell.conected():
            self.shell = ssh(host='192.168.16.2', user='pi', password='raspberry')
        if self.controller.poll() is not None:  # poll gives none is it's still running
            self.controller = self.shell.process(['python3', robo_pwn_controller],
                                                 cwd="/home/pi/Desktop/newCapstone/middleware/devices")
        print('connected')
    def exec_control_command(self, command):
        self.connect()
        match command:
            case "f":
                self.controller.sendline(b"f 0.25")
                self.controller.recvuntil(b'> ')
                return
            case "l":
                self.controller.sendline(b"f 0.25")
                self.controller.recvuntil(b'> ')
                self.controller.sendline(b"l 0.55")
                self.controller.recvuntil(b'> ')
                return
            case "r":
                self.controller.sendline(b"f 0.25")
                self.controller.recvuntil(b'> ')
                self.controller.sendline(b"r 0.55")
                self.controller.recvuntil(b'> ')
                return

    def get_last_pic(self):
        self.connect()
        gotPhoto = False
        while not gotPhoto:
            try:
                self.shell.download_file('/tmp/pics/lastPic','mostRecentPhoto')
                gotPhoto = True
            except:
                pass

def testRawCommand():
    connection = Connection()
    connection.exec_control_command("f")

if __name__ == "__main__":
    testRawCommand()