
# Robot Setup

## Github Links and Additional Resources:
### Camera
- [https://github.com/jacksonliam/mjpg-streamer](https://github.com/jacksonliam/mjpg-streamer)  
- https://github.com/inactivitytimeout/labists_pi_car/blob/main/Technical.md


# Usage Instructions

## Connection:
### Connecting to the wifi:
- Power the robot on
- Look for the wifi network “RPiRobot2” and hit connect. If you don’t see the network, give the robot a minute or two to start up
- If prompted to enter a pin off the label, select connect with security key instead
- Password is “password123”
  
### Connecting to the command prompt:
- You must first be connected to the wifi
- In command prompt enter “ssh pi@192.168.16.2”
- When prompted, the password is “raspberry”

### Connecting to FTP:
- These instructions use WinSCP, but any FTP software should work
- Host name: 192.168.16.2 ; Username:pi ; Password: raspberry

### Connecting via ethernet:

-   
    

## Camera:

### Starting the video web server:
* Navigate to Home Folder “cd $HOME” or “cd ~”
- sudo LD_LIBRARY_PATH=’pwd’ ./mjpg_streamer -i "./input_uvc.so -d /dev/video0" -o "./output_http.so -p 8080 -w ./www" & On the PC: http://<car_ip_address>:8080/javascript.html
- If an output file is needed or input is changed, either can be changed according to Liam Jackson’s file.
- For example, to get jpg images: sudo LD_LIBRARY_PATH=’pwd’ ./mjpg_streamer -i "./input_uvc.so -d /dev/video0" -o "./output_file.so”
- *Jpg images will go in the ~/tmp folder
- If ./mjpg_streamer is not autofilling
- “cd mjpg-streamer/mjpg-streamer-experimental/”
- Run the command “make”
- If you get the error “sudo: ./mjpg_streamer: command not found”
- Stop the process with ^c
- “cd mjpg-streamer/mjpg-streamer-experimental/”
- Run the command without setting the PATH: sudo ./mjpg_streamer -i "./input_uvc.so -d /dev/video0" -o "./output_http.so -p 8080 -w ./www" & On the PC: http://192.168.16.2:8080/javascript.html

### Input plugins:
- input_file
- input_http
- input_opencv (documentation)
- input_ptp2
- input_raspicam (documentation)
- input_uvc (documentation)

### Output plugins:
- output_file  
- output_http (documentation)
- output_rtsp (not functional)
- output_udp (not functional)
- output_viewer (documentation)
- output_zmqserver (documentation)

## Movement Control
- Start by navigating to the controller.py file
	- ~/Desktop/newCapstone/middleware/devices/
- Start controller.py with “python3 controller.py”

### Drive Commands
- f <runtime> : forward for given runtime, if not provided, runtime = 0.28 seconds
- l <runtime> : left turn for given runtime, if not provided, runtime = 0.28 seconds
- r <runtime> : right turn for given runtime, if not provided, runtime = 0.28 seconds
- b <runtime> :  backwards for given runtime, if not provided, runtime = 0.28 seconds  

### Chassis Control Commands
- lights <T/F> : turn headlights on or off
- Servo <angle in degrees> : rotate the camera up or down to position  

### Additional Information
- Runtime in seconds float
- Servo angle is 90 default
