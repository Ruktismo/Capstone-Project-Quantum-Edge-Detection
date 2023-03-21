# Importing standard Qiskit libraries and configuring account
# Libs needed: qiskit, matplotlib, pylatexenc, qiskit-ibm-runtime
from qiskit import *
from qiskit.compiler import transpile
from qiskit.providers.fake_provider.backends.guadalupe.fake_guadalupe import FakeGuadalupeV2 # for 16x16
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler

#standard libraries needed
import sys
import time
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


    fake_backend = FakeGuadalupeV2()

    # Transpile the circuits for optimized execution on the backend
    # We made the circuits with high-level gates, need to decompose to basic gates so IBMQ hardware can understand
    qc_small_h_t = transpile(qc_h, fake_backend, optimization_level=3)
    qc_small_v_t = transpile(qc_v, fake_backend, optimization_level=3)

    # Combining both circuits into a list
    circ_list_t = [qc_small_h_t, qc_small_v_t]

    #to calculate compilation time then output to console
    tok = time.perf_counter()
    t = tok - tic
    print(f"Total Compile Time: {t:0.4f} seconds")

    #connect to simulator
    #using QASM_SIMULATOR
    service = QiskitRuntimeService(channel="ibm_quantum", token=TOKEN)
    # Set backend to "ibmq_qasm_simulator" for non-quantum results, for quantum results use "ibmq_belem" or other
    with Session(service=service, backend="ibmq_qasm_simulator") as session:
        sampler = Sampler(session=session)
        job = sampler.run(circ_list_t, shots=8192)
        print("\nJob queued, look to IBM website to get time updates.\nDO NOT CLOSE PROGRAM!!!")
        # Getting the resultant probability distribution after measurement
        result = job.result()  # Blocking until IBM returns with results


    counts_h = {f'{k:0{total_qb}b}': 0.0 for k in range(2 ** total_qb)}
    counts_v = {f'{k:0{total_qb}b}': 0.0 for k in range(2 ** total_qb)}

    for k, v in result.quasi_dists[0].items():
        counts_h[format(k, f"0{total_qb}b")] = v
    for k, v in result.quasi_dists[1].items():
        counts_v[format(k, f"0{total_qb}b")] = v


    # Extracting counts for odd-numbered states. i.e. data that we are interested in
    edge_scan_h = np.array([counts_h[f'{2 * i + 1:0{total_qb}b}'] for i in range(2 ** data_qb)]).reshape(16, 16)
    edge_scan_v = np.array([counts_v[f'{2 * i + 1:0{total_qb}b}'] for i in range(2 ** data_qb)]).reshape(16, 16).T

    edge_scan_sim = edge_scan_h + edge_scan_v

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
    print('running local 16x16 simulation...')
    print('*************************************************')

    local16x16()

if __name__ == "__main__":
    main()