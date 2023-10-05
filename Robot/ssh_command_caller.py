# Python script to execute commands over SSH connection
# requires paramiko
# FOR USE WITH CAPSTONE ROBOT: make sure to connect to robot's Wi-Fi before using.
# Wifi Information - Name: RPiRobot2, Pass:  password123
import paramiko

# ==================== Server Credentials and Saved Globals ==================== #
DEFAULT_PORT = "22"

robo_server = "192.168.16.2"
robo_user = "pi"
robo_password = "raspberry"

robo_control_path = "Desktop/newCapstone/middleware/devices/"
robo_control_base = "python3 controller.py"


# ==================== SSH Control Commands ==================== #
class Connection:
    ssh = paramiko.SSHClient()

    def connect(self, server, user, password, port):
        self.ssh.set_missing_host_key_policy(paramiko.client.AutoAddPolicy)  # allow to connect to unknown servers

        # connect using specified or default port
        if port != "":
            self.ssh.connect(server, username=user, password=password, port=port)
        else:
            self.ssh.connect(server, username=user, password=password, port=DEFAULT_PORT)

    def prep_start(self, path):
        self.ssh.exec_command("cd ~")
        self.ssh.exec_command("cd " + path)

    def pass_command(self, command):
        return self.ssh.exec_command(command)[1]

    def close(self):
        self.ssh.close


# ==================== Test operations ==================== #
def test():
    # ===== create connection object
    ssh_connection = Connection()

    forward = robo_control_base + "-c f -t 0"
    backwards = robo_control_base + "-c b -t 0"
    left = robo_control_base + "-c l -t 0"
    right = robo_control_base + "-c r -t 0"
    stop = robo_control_base + "-c s -t 0"
    servo = robo_control_base + "-c servo -t 0"
    lights = robo_control_base + "-c lights -t 0"

    # ===== robo drive control test
    ssh_connection.connect(server=robo_server, username=robo_user, password=robo_password)
    ssh_connection.prep_start(path=robo_control_path)
    ssh_connection.pass_command(command=forward)
    ssh_connection.pass_command(command=backwards)
    ssh_connection.pass_command(command=left)
    ssh_connection.pass_command(command=backwards)
    ssh_connection.pass_command(command=right)

    # ===== close the connection
    ssh_connection.close()


if __name__ == '__main__':
    test()
