# Python script to execute commands over SSH connection
# requires paramiko
# FOR USE WITH CAPSTONE ROBOT: make sure to connect to robot's Wi-Fi before using.
# Wifi Information - Name: RPiRobot2, Pass:  password123
import paramiko

# ==================== Server Credentials and Saved Globals ==================== #
robo_server = "192.168.16.2"
robo_port = 22
robo_user = "pi"
robo_password = "raspberry"
fps = 12

robo_control_path = "Desktop/newCapstone/middleware/devices/"
robo_control_base = "python3 controller.py"
robo_photo_path = "/tmp/pics"
robo_last_photo = "lastPic"
robo_cam_cmd = f"sudo LD_LIBRARY_PATH='pwd' ./mjpg_streamer -i \"input_uvc.so -d /dev/video0 -f {fps}\" -o \"./output_file.so -f {robo_photo_path} -l {robo_last_photo}\""

robo_forward = robo_control_base + " --command f --runtime 0"
robo_backwards = robo_control_base + " --command b --runtime 0"
robo_left = robo_control_base + " --command l --runtime 0"
robo_right = robo_control_base + " --command r --runtime 0"
robo_stop = robo_control_base + " --command s --runtime 0"
robo_servo = robo_control_base + " --command servo --runtime 0"
robo_lights = robo_control_base + " --command lights --runtime 0"


# ==================== SSH Control Commands ==================== #
class Connection:
    ssh = paramiko.SSHClient()

    def connect(self):
        self.ssh.set_missing_host_key_policy(paramiko.client.AutoAddPolicy)  # allow to connect to unknown servers

        # connect using specified or default port
        self.ssh.connect(robo_server, username=robo_user, password=robo_password, port=robo_port)

        self.sftp = self.ssh.open_sftp()

        try:
            self.sftp.chdir(robo_photo_path)  # Test if remote_path exists
        except IOError:
            self.sftp.mkdir(robo_photo_path)  # Create remote_path
            self.sftp.chdir(robo_photo_path)

        self.ssh.exec_command(f"cd /tmp/pics ; rm ./*;")
        self.ssh.exec_command(f"cd ~/mjpg-streamer/mjpg-streamer-experimental/ ; {robo_cam_cmd}")

    def exec_control_command(self, command):
        match command:
            case "f":
                return self.ssh.exec_command(f"cd ~ ; cd {robo_control_path} ; {robo_forward}")
            case "b":
                return self.ssh.exec_command(f"cd ~ ; cd {robo_control_path} ; {robo_backwards}")
            case "l":
                return self.ssh.exec_command(f"cd ~ ; cd {robo_control_path} ; {robo_left}")
            case "r":
                return self.ssh.exec_command(f"cd ~ ; cd {robo_control_path} ; {robo_right}")
            case "s":
                return self.ssh.exec_command(f"cd ~ ; cd {robo_control_path} ; {robo_stop}")
            case "servo":
                return self.ssh.exec_command(f"cd ~ ; cd {robo_control_path} ; {robo_servo}")
            case "lights":
                return self.ssh.exec_command(f"cd ~ ; cd {robo_control_path} ; {robo_lights}")

    def get_last_pic(self):
        self.sftp.get(robo_last_photo, "mostRecentPhoto")