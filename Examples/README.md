<h1> Quantum Edge Detection Examples </h1>

Background Knowledge on Qiskit:
https://qiskit.org/textbook/preface.html

Example Explanation of QED for smaller and larger images
(Image example for testing also taken):
https://qiskit.org/textbook/ch-applications/quantum-edge-detection.html

IBM account to get token:
https://quantum-computing.ibm.com/

For constructing larger decrement gates guide:
https://algassert.com/circuits/2015/06/12/Constructing-Large-Increment-Gates.html


-2x2hardware.py:
    -runs on real quantum hardware, so queue time may be very long
    -use IBM token in account (copy from clipboard, do not ctrl+c)
-8x8local.py:
    -runs on local machine as a simulator
    -uses state vector simulator
    -uses https://qiskit.org/textbook/ch-applications/quantum-edge-detection.html#Quantum-Probability-Image-Encoding-(QPIE)
-16x16sim.py:
    -expansion of 8x8 simulation
-Examples.py:
    -combination of all 3 of above examples in one file
    -menu to select which option to run

Notes:
-nxn array manually created to run the edge detection using 0's and 1's.
-1 pixel of "edge" will be off
    -no way to fix. because of edge detection running horizontally then vertically, or vise versa,
        will always be off by one pixel. the way to fix is VERY complicated
    -reference color intensity differences comparing to adjacent pixels
    -qasm simulator can handle intensities, state vector does not.