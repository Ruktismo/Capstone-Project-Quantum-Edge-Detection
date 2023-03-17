<h1>Capstone Project: Quantum Edge Detection</h1>

- Andrew Ericksson
- Yumi Lamansky
- Meredith Kuhler
- Michael Del Regno
- Kenneth Wang
- Zaid Buni

https://github.com/jacksonliam/mjpg-streamer
This was used to get camera output, but no commands from that page are needed instead here are the necessary commands if the files are already properly installed.

Here are the Current Instructions:
Starting the video web server:

sudo LD_LIBRARY_PATH=`pwd` ./mjpg_streamer -i "./input_uvc.so -d /dev/video0" -o "./output_http.so -p 8080 -w ./www" &
On the PC: http://<car_ip_address>:8080/javascript.html

