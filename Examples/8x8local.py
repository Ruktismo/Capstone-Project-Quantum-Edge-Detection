# Importing standard Qiskit libraries and configuring account
# Libs needed: qiskit, matplotlib, pylatexenc, qiskit-ibm-runtime
from qiskit import *

#standard libraries needed
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

#error check for command args being passed in
try:
    TOKEN = sys.argv[1]
except IndexError:
    print(f"ERROR: INCORRECT NUMBER OF ARGS")
    print(f"Expected: [Token,H-Size,V-Size]\nGot: {sys.argv}")
    exit()

# Function for plotting the image using matplotlib
# parameters: image and title
def plot_image(img, title: str):
    plt.title(title)    #display the string title on the image
    plt.xticks(range(img.shape[0])) #display ticks on x axis
    plt.yticks(range(img.shape[1])) #display ticks on y axis

    #.imshow is built in function of plot library
    plt.imshow(img, extent=[0, img.shape[0], img.shape[1], 0], cmap='viridis')
    # A blocking request to display the figure(s) loaded. Block ends when user closes figure(s)
    # Will show glitchy overlap if mutable figures are made before show is called
    plt.show()


# Convert the raw pixel values to probability amplitudes
# i.e. sum of all pixels is one
def amplitude_encode(img_data):
    # Calculate the RMS value
    rms = np.sqrt(np.sum(np.sum(img_data ** 2, axis=1)))  # sum up all pixels to get total

    # Create normalized image
    image_norm = []
    for arr in img_data:
        for ele in arr:
            image_norm.append(ele / rms)  # divide pixel by total to get probability

    # Return the normalized image as a numpy array
    return np.array(image_norm)





"""
8x8 example made from Quskit docs:
    https://qiskit.org/textbook/ch-applications/quantum-edge-detection.html#Quantum-Probability-Image-Encoding-(QPIE)

Had to change a few things to get it working on local python since jupyter has some built in functions we don't have.
"""
def local8x8():
    style.use('bmh')  # This is setting the color scheme for the plots.

    # A 8x8 binary image represented as a numpy array
    image = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 1, 1, 1, 0, 0],
                      [0, 1, 1, 1, 1, 1, 1, 0],
                      [0, 1, 1, 1, 1, 1, 1, 0],
                      [0, 1, 1, 1, 1, 1, 1, 0],
                      [0, 0, 0, 1, 1, 1, 1, 0],
                      [0, 0, 0, 1, 1, 1, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0]])

    # Get the amplitude ancoded pixel values
    # Horizontal: Original image
    image_norm_h = amplitude_encode(image)

    # Vertical: Transpose of Original image
    image_norm_v = amplitude_encode(image.T)

    # Initialize some global variable for number of qubits
    data_qb = 6  # Set to ceil(log_2(image.size)) hardcoded as 6 since image does not change
    anc_qb = 1  # This is most likely the auxiliary qbit. But I'm not sure since I don't understand the name "anc"?
    total_qb = data_qb + anc_qb

    # Initialize the amplitude permutation unitary
    # i.e. make identity matrix and shift all indexes right one with rollover
    D2n_1 = np.roll(np.identity(2 ** total_qb), 1, axis=1)

    # Create the circuit for horizontal scan
    # PyCharm gives a warning that methods "initialize" and "unitary" do not exist. It is ok they are made at runtime.
    qc_h = QuantumCircuit(total_qb)
    qc_h.initialize(image_norm_h, range(1, total_qb))
    qc_h.h(0)
    qc_h.unitary(D2n_1, range(total_qb))
    qc_h.h(0)
    qc_h.draw('mpl', fold=-1)  # Make render of circuit in matplotlib
    plt.show()  # QuantumCircuit.draw makes the figures but does not display

    # Create the circuit for vertical scan
    qc_v = QuantumCircuit(total_qb)
    qc_v.initialize(image_norm_v, range(1, total_qb))
    qc_v.h(0)
    qc_v.unitary(D2n_1, range(total_qb))
    qc_v.h(0)

    ##.DRAW RETURNS FIGURE; into a var?
    qc_v.draw('mpl', fold=-1)
    plt.show()

    # Combine both circuits into a single list
    circ_list = [qc_h, qc_v]

    # Simulating the circuits
    back = Aer.get_backend('statevector_simulator')  # Get async background process to run sim
    results = execute(circ_list, backend=back).result()  # Run circuits on backend. Block to get results
    sv_h = results.get_statevector(qc_h)
    sv_v = results.get_statevector(qc_v)

    # Classical postprocessing for plotting the output

    # Defining a lambda function for thresholding to binary values
    # This can be helpful for smoothing out quantum randomness.
    threshold = lambda amp: (amp > 1e-15 or amp < -1e-15)  # Take with caution can destroy/muddle data if set wrong.

    # Selecting odd states from the raw statevector and reshaping column vector of size 64 to an 8x8 matrix
    edge_scan_h = np.abs(np.array([1 if threshold(sv_h[2 * i + 1].real) else 0 for i in range(2 ** data_qb)])).reshape(
        8, 8)
    edge_scan_v = np.abs(np.array([1 if threshold(sv_v[2 * i + 1].real) else 0 for i in range(2 ** data_qb)])).reshape(
        8, 8).T

    # Combining the horizontal and vertical component of the result
    edge_scan_sim = edge_scan_h | edge_scan_v

    #combine all images
    #create base
    fig, imageAxis = plt.subplots(2, 2)

    #display each image
    imageAxis[0, 0].imshow(image)
    imageAxis[0, 1].imshow(edge_scan_h)
    imageAxis[1, 0].imshow(edge_scan_v)
    imageAxis[1, 1].imshow(edge_scan_sim)

    #display titles for subplots
    imageAxis[0, 0].set_title('Original')
    imageAxis[0, 1].set_title('Horizontal Scan')
    imageAxis[1, 0].set_title('Vertical Scan')
    imageAxis[1, 1].set_title('Edge Detected Image')

    #adjust the spacing between images
    plt.subplots_adjust(hspace = 0.5, wspace = 0.75)

    #Show/Display
    plt.show()


def main():
    print('*************************************************')
    print('running local 8x8 simulation...')
    print('*************************************************')

    local8x8()

if __name__ == "__main__":
    main()