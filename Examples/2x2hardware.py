# Importing standard Qiskit libraries and configuring account
# Libs needed: qiskit, matplotlib, pylatexenc, qiskit-ibm-runtime
from qiskit import *
from qiskit.compiler import transpile
from qiskit.providers.fake_provider.backends.belem.fake_belem import FakeBelemV2 # for 2x2
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
from qiskit.visualization import plot_histogram

#standard libraries needed
import sys
import numpy as np
import matplotlib.pyplot as plt

#error check for command args being passed in
#TOKEN will be IBM quantum account API token
#be sure to copy from clipboard (NOT the keyboard shortcut)
try:
    TOKEN = sys.argv[1]
except IndexError:
    print(f"ERROR: INCORRECT NUMBER OF ARGS")
    print(f"Expected: [Token]\nGot: {sys.argv}")
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
    #circ_list = [qc_small_h, qc_small_v]  #this is not used but keeping here if need in future

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
    print('*************************************************')
    print('2x2 real hardware')
    print('*************************************************')
    input("\nNote: Running on real hardware takes a long queue time, hours long depending on availability."
          "\nIf you want to end the experiment you have to go to the IBM website and cancel the job."
          "\nHit enter to confirm your selection.")

    hardware2x2()

if __name__ == "__main__":
    main()