# Importing standard Qiskit libraries and configuring account
# Libs needed: qiskit, matplotlib, pylatexenc, qiskit-ibm-runtime
from qiskit import QuantumCircuit, execute, Aer

#standard libraries needed
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


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


# 16x16 image simulation
def local16x16():
    style.use('bmh')  # color scheme

    # hardcoded 16x16 binary image represented as a numpy array. 0 will be dark/off, 1 will be light/on
    # this particular one sort of looks like a smiley face :)
    image = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                      [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                      [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                      [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    tic = time.perf_counter()
    # Get horizontal and vertical amplitudes encoded pixel values
    image_norm_h = amplitude_encode(image)
    image_norm_v = amplitude_encode(image.T)  # Image transpose for the vertical

    # n = log2 N
    # for qubits
    data_qb = 8
    anc_qb = 1  # Aux qbit
    total_qb = data_qb + anc_qb

    # Create the circuit for horizontal scan; built-in functions
    qc_h = QuantumCircuit(total_qb)
    qc_h.initialize(image_norm_h, range(1, total_qb))
    qc_h.barrier()
    qc_h.h(0)
    qc_h.barrier()

    # Decrement gate - START
    qc_h.x(0)
    qc_h.cx(0, 1)
    qc_h.ccx(0, 1, 2)
    for c in range(3, total_qb):
        qc_h.mcx([b for b in range(c)], c)
    # Decrement gate - END

    qc_h.barrier()
    qc_h.h(0)
    qc_h.measure_all()


    # Create the circuit for vertical scan; built in functions
    qc_v = QuantumCircuit(total_qb)
    qc_v.initialize(image_norm_v, range(1, total_qb))
    qc_v.barrier()
    qc_v.h(0)
    qc_v.barrier()

    # Decrement gate - START
    qc_v.x(0)
    qc_v.cx(0, 1)
    qc_v.ccx(0, 1, 2)
    for c in range(3, total_qb):
        qc_v.mcx([b for b in range(c)], c)
    # Decrement gate - END

    qc_v.barrier()
    qc_v.h(0)
    qc_v.measure_all()


    # Combine both circuits into a single list
    circ_list = [qc_h, qc_v]
    # to calculate compilation time then output to console
    tok = time.perf_counter()
    t = tok - tic
    print(f"Total Compile Time: {t:0.4f} seconds")


    #using QASM_SIMULATOR
    tic = time.perf_counter()

    backend = Aer.get_backend('qasm_simulator')
    job = execute(circ_list, backend, shots=1024)
    result = job.result().get_counts()
    # Measuring run time
    tok = time.perf_counter()
    t = tok - tic
    print(f"Total Run Time: {t:0.4f} seconds")


    # Make dic with keys that are binaries from 0 to 2^total_qb. with values of 0.0 to start.
    # This is done since IBM will not return Qbit configs that are 0. So we need to map the results to the full space
    # Formatted String: 0 (for padding zeros) {total_qb} (for bit-string size) b (to format from int to binary)
    counts_h = {f'{k:0{total_qb}b}': 0.0 for k in range(2 ** total_qb)}
    counts_v = {f'{k:0{total_qb}b}': 0.0 for k in range(2 ** total_qb)}

    # Transfer all known values form experiment results to dic
    for k, v in result[0].items():
        counts_h[k] = v / 1024  # div by 1024 since v is the counts, and we need a probability between 0-1
    for k, v in result[1].items():
        counts_v[k] = v / 1024


    # Extracting counts for odd-numbered states. i.e. data that we are interested in
    edge_scan_h = np.array([counts_h[f'{2 * i + 1:0{total_qb}b}'] for i in range(2 ** data_qb)]).reshape(16, 16)  #horizontal
    edge_scan_v = np.array([counts_v[f'{2 * i + 1:0{total_qb}b}'] for i in range(2 ** data_qb)]).reshape(16, 16).T  #transpose for vertical

    # Combine H and V scans to get full edge detection image
    edge_scan_sim = edge_scan_h + edge_scan_v

    #combine all images
    # create 2x2 base to fit 1 image in each quadrant
    fig, imageAxis = plt.subplots(2, 2)

    #display each image on the axis
    imageAxis[0, 0].imshow(image) #top left
    imageAxis[0, 1].imshow(edge_scan_h)  #top right
    imageAxis[1, 0].imshow(edge_scan_v)  #bottom left
    imageAxis[1, 1].imshow(edge_scan_sim)  #bottom right

    #display titles for subplots
    imageAxis[0, 0].set_title('Original')  #top left
    imageAxis[0, 1].set_title('Horizontal Scan')  #top right
    imageAxis[1, 0].set_title('Vertical Scan')  #bottom left
    imageAxis[1, 1].set_title('Edge Detected Image')  #bottom right

    #adjust the spacing between images
    plt.subplots_adjust(hspace = 0.5, wspace = 0.75)

    #Show/Display
    plt.show()

# MAIN
def main():
    print('*************************************************')
    print('running local 16x16 simulation...')
    print('*************************************************')

    local16x16()

if __name__ == "__main__":
    main()