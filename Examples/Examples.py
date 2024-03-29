# Importing standard Qiskit libraries and configuring account
# Libs needed: qiskit, matplotlib, pylatexenc, qiskit-ibm-runtime
from qiskit import *
from qiskit.compiler import transpile
from qiskit.providers.fake_provider.backends.belem.fake_belem import FakeBelemV2 # for 2x2
from qiskit.providers.fake_provider.backends.guadalupe.fake_guadalupe import FakeGuadalupeV2 # for 16x16
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
from qiskit.visualization import plot_histogram

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
# i.e. sum of all pixels is 1
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


    # Create the circuit for vertical scan; built-in functions
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


"""
8x8 example made from Qiskit docs:
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

    # Get the amplitude encoded pixel values
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

    # Selecting odd states from the raw state-vector and reshaping column vector of size 64 to a 8x8 matrix
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



"""
if running hardware for the first time make sure to run the following cmds in a python console.
    import matplotlib
    from qiskit_ibm_runtime import QiskitRuntimeService

    QiskitRuntimeService.save_account(channel="ibm_quantum", token="MY_IBM_QUANTUM_TOKEN")

Still need to figure out how the circuit is designed.
The img is not directly used, but it works? 
"""
def hardware2x2():
    # Create a 2x2 image to be run on the hardware
    image_small = np.array([[0, 1],
                            [0, 0]])

    # Plotting the image_small using matplotlib
    plot_image(image_small, 'Cropped image')

    # Get the amplitude encoded pixel values
    # Horizontal: Original image
    image_norm_h = amplitude_encode(image_small)

    # Vertical: Transpose of Original image
    image_norm_v = amplitude_encode(image_small.T)

    # Initialize the number of qubits
    data_qb = 2
    anc_qb = 1
    total_qb = data_qb + anc_qb

    # Create the circuit for horizontal scan
    qc_small_h = QuantumCircuit(total_qb)
    qc_small_h.initialize(image_norm_h, range(1, total_qb))
    qc_small_h.x(1)  # apply XGate to qbit 1
    qc_small_h.h(0)  # apply hadamard gate to qbit 0

    # Decrement gate - START
    qc_small_h.x(0)
    qc_small_h.cx(0, 1)
    qc_small_h.ccx(0, 1, 2)
    # Decrement gate - END

    qc_small_h.h(0)
    qc_small_h.measure_all()
    qc_small_h.draw('mpl')
    plt.show()

    # Create the circuit for vertical scan
    qc_small_v = QuantumCircuit(total_qb)
    qc_small_v.initialize(image_norm_v, range(1, total_qb))
    qc_small_v.x(2)
    qc_small_v.h(0)

    # Decrement gate - START
    qc_small_v.x(0)
    qc_small_v.cx(0, 1)
    qc_small_v.ccx(0, 1, 2)
    # Decrement gate - END

    qc_small_v.h(0)
    qc_small_v.measure_all()
    qc_small_v.draw('mpl')
    plt.show()

    # Combine both circuits into a single list
    circ_list = [qc_small_h, qc_small_v]

    # Fake Backend for transpile to decompose circuit to. Locked to Belem for now
    # Each quantum computer supports different gates, transpile needs to know what gates are available
    fake_backend = FakeBelemV2()  # TODO: Allow user to set backend so they can run on any quantum computer.

    # Transpile the circuits for optimized execution on the backend
    # We made the circuits with high-level gates, need to decompose to basic gates so IBMQ hardware can understand
    qc_small_h_t = transpile(qc_small_h, fake_backend, optimization_level=3)
    qc_small_v_t = transpile(qc_small_v, fake_backend, optimization_level=3)

    # Combining both circuits into a list
    circ_list_t = [qc_small_h_t, qc_small_v_t]

    # display transpiled circuits
    circ_list_t[0].draw('mpl', fold=-1)
    plt.show()
    circ_list_t[1].draw('mpl', fold=-1)
    plt.show()

    # Load the IBMQ account. It is not properly loading, but it is saved?
    # IBMQ.load_account()
    # Using token from arg[1] since load_account is not working
    service = QiskitRuntimeService(channel="ibm_quantum", token=TOKEN)
    # Make a new session with IBM
    # Set backend to "ibmq_qasm_simulator" for non-quantum results, for quantum results use "ibmq_belem" or other
    with Session(service=service, backend="ibmq_belem") as session:
        sampler = Sampler(session=session)  # Make a Sampler to run the circuits
        # Executing the circuits on the backend
        job = sampler.run(circ_list_t, shots=8192)
        # job_monitor(job)  # job_monitor does not work
        print("\nJob queued, look to IBM website to get time updates.\nDO NOT CLOSE PROGRAM!!!")
        # Getting the resultant probability distribution after measurement
        result = job.result()  # Blocking until IBM returns with results

    # Make dic with keys that are binaries from 0 to 2^total_qb. with values of 0.0 to start.
    # This is done since IBM will not return Qbit configs that are 0. So we need to map the results to the full space
    # Formatted String: 0 (for padding zeros) {total_qb} (for bit-string size) b (to format from int to binary)
    counts_h = {f'{k:0{total_qb}b}': 0.0 for k in range(2 ** total_qb)}
    counts_v = {f'{k:0{total_qb}b}': 0.0 for k in range(2 ** total_qb)}
    # Transfer all known values form experiment results to dic
    for k, v in result.quasi_dists[0].items():
        counts_h[format(k, f"0{total_qb}b")] = v
    for k, v in result.quasi_dists[1].items():
        counts_v[format(k, f"0{total_qb}b")] = v

    print('Counts for Horizontal scan:')
    plot_histogram(counts_h)
    plt.show()

    print('\n\nCounts for Vertical scan:')
    plot_histogram(counts_v)
    plt.show()

    # Extracting counts for odd-numbered states. i.e. data that we are interested in
    edge_scan_small_h = np.array([counts_h[f'{2 * i + 1:03b}'] for i in range(2 ** data_qb)]).reshape(2, 2)
    edge_scan_small_v = np.array([counts_v[f'{2 * i + 1:03b}'] for i in range(2 ** data_qb)]).reshape(2, 2).T

    plot_image(edge_scan_small_h, 'Horizontal scan output')
    plot_image(edge_scan_small_v, 'Vertical scan output')

    # Combine H and V scans to get full edge detection image
    edge_detected_image_small = edge_scan_small_h + edge_scan_small_v

    # Plotting the original and edge-detected images
    plot_image(edge_detected_image_small, 'Full Edge Detected Image')


def main():
    #get input from user for selection
    sim_choice = int(input("1) 8x8 local\n"
                           "2) 16x16 local\n"
                           "3) 2x2 real-hardware\n"
                            "\nWhat simulation to run?\t"))
    #run 8x8 if 1
    if sim_choice == 1:
        local8x8()
    #run 16x16 if 2
    elif sim_choice == 2:
        local16x16()
    #run 2x2 hardware if 3
    elif sim_choice == 3:
        input("\nNote: Running on real hardware takes a long queue time, hours long depending on availability."
              "\nIf you want to end the experiment you have to go to the IBM website and cancel the job."
              "\nHit enter to confirm your selection.")
        hardware2x2()
    else:
        print("invalid choice")


if __name__ == "__main__":
    main()

"""
NOTES:
print(f"{dir(obj)}") use dir on an obj to get what methods it has access to at runtime
"""
