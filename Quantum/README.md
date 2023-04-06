<h1>Capstone Project: Quantum Edge Detection</h1>

Implement quantum edge detection (QED) for self-driving of robot car.

- Andrew Ericksson
- Yumi Lamansky
- Meredith Kuhler
- Michael Del Regno
- Kenneth Wang
- Zaid Buni

Previous capstone team resources:
-https://github.com/zerohezitation/Capstone-Project-AI-Robot-Car-Maze-Navigation/
-https://github.com/zerohezitation/Capstone-Project-AI-Robot-Car-Maze-Navigation/commit/3d2fe0142d027573772e881a9d1f3cd0eaafb32e#diff-7e55c279e89f81befe625cf676baad0f71060a0b3f950698055d2b16bcd7d344


Background Knowledge on Qiskit:
https://qiskit.org/textbook/preface.html

Example Explanation of QED for smaller and larger images
(Image example for testing also taken):
https://qiskit.org/textbook/ch-applications/quantum-edge-detection.html

IBM account to get token: (free sign up)
https://quantum-computing.ibm.com/

For constructing larger decrement gates guide:
https://algassert.com/circuits/2015/06/12/Constructing-Large-Increment-Gates.html

-split 256x256 into 16x16 "chunks" and run QED on each chunk
-reduced shots from 8192 to 1024 for efficiency
-ccx decrement gate for complexity issue??

Current roadblocks:
-plug everything (robot + quantum) together
-run time issues
    -prefer work on order of seconds
    -number of gates executed affect runtime
-chunk anomaly/artifacts
    -because scanning in black and white instead of grayscale?
    -if in B&W, pixels can be combination so edge detection is angry
    -check B&W version that it's scanning
    -go into grayscale-lambda function and pick range for values
-for efficiency:
    -real time: state vector faster
    -qasm average could take ~1 hour (is normal) for real quantum (can take days)
    -smaller images better
