import numpy
from qiskit import *
from qiskit.compiler import transpile
from qiskit.providers.fake_provider.backends.belem.fake_belem import FakeBelemV2
from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler

import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import style

# error check for cmd args
try:
    TOKEN = sys.argv[1]
    H_SIZE = int(sys.argv[2])
    V_SIZE = int(sys.argv[3])
except IndexError:
    print(f"ERROR: INCORRECT NUMBER OF ARGS\nExpected: [Token,H-Size,V-Size]\nGot: {sys.argv}")
    exit()
except ValueError:
    print("ERROR: CROP SIZE NOT OF TYPE INT\n\tUsing default of 16x16")
    H_SIZE = 16
    V_SIZE = 16

# Function for plotting the image using matplotlib
def plot_image(img, title: str):
    plt.title(title)
    plt.xticks(range(img.shape[0]))
    plt.yticks(range(img.shape[1]))
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


def crop(image, hsize, vsize):
    h_chunks = image.size[0] / hsize
    v_chunks = image.size[1] / vsize
    # quick error check for hsplit and vsplit
    if (image.size[0] % hsize != 0) or (image.size[1] % vsize != 0):
        print("ERROR\nImage is not cleanly dividable by chunk size.")
    # Split the image vertically then pump all vertical slices into hsplit to get square chunks
    nested_chunks = [np.hsplit(vs, h_chunks) for vs in np.vsplit(image, v_chunks)]
    # The split process leaves us with the chunks in a nested array, flatten to a list
    img_chunks = [item for sublist in nested_chunks for item in sublist]
    return img_chunks

# TODO this ia a copy of the 2x2. need to update to 16x16
def circuit_h(img, total_qb):
    # Create the circuit for horizontal scan
    qc_small_h = QuantumCircuit(total_qb)
    qc_small_h.initialize(img, range(1, total_qb))
    qc_small_h.x(1)  # apply XGate to qbit 1
    qc_small_h.h(0)  # apply hadamard gate to qbit 0

    # Decrement gate - START
    qc_small_h.x(0)
    qc_small_h.cx(0, 1)
    qc_small_h.ccx(0, 1, 2)

    # Decrement gate - END
    qc_small_h.h(0)
    qc_small_h.measure_all()
    return qc_small_h


def circuit_v(img, total_qb):
    # Create the circuit for vertical scan
    qc_small_v = QuantumCircuit(total_qb)
    qc_small_v.initialize(img, range(1, total_qb))
    qc_small_v.x(2)
    qc_small_v.h(0)

    # Decrement gate - START
    qc_small_v.x(0)
    qc_small_v.cx(0, 1)
    qc_small_v.ccx(0, 1, 2)
    # Decrement gate - END

    qc_small_v.h(0)
    qc_small_v.measure_all()
    return qc_small_v

#####################################################
def sim256x256():
    style.use('bmh')  # This is setting the color scheme for the plots.
    # TODO add image path
    pic = Image.open("./edge_detection_input.jpg").crop((0, 0, 255, 255))  # open image and crop to 256x256
    image = numpy.asarray(pic)

    plot_image(image, 'Original Image')

    croped_imgs = crop(image, H_SIZE, V_SIZE)
    # Get the amplitude ancoded pixel values
    # Horizontal: Original image
    image_norms_h = []
    for chunkH in croped_imgs:
        image_norms_h.append(amplitude_encode(chunkH))

    # Vertical: Transpose of Original image
    image_norms_v = []
    for chunckV in croped_imgs:
        image_norms_v.append(amplitude_encode(chunckV.T))

    #todo: print the image_norms v and h here to see
    print(image_norms_h)
    print(image_norms_v)



    # Initialize some global variable for number of qubits
    data_qb = 8  # Set to ceil(log_2(image.CropSize)) hardcoded as 8 since image crop is 16x16
    anc_qb = 1  # This is the auxiliary qbit.
    total_qb = data_qb + anc_qb

    # make a circuit for each horizontal and vertical chunk
    circuits_h = []
    for img in image_norms_h:
        circuits_h.append(circuit_h(img, total_qb))

    circuits_v = []
    for img in image_norms_v:
        circuits_v.append(circuit_v(img, total_qb))

    # combine all circuits into one list
    circuit_list = circuits_h
    circuit_list.extend(circuits_v)

    # Fake Backend for transpile to decompose circuit to. Locked to Belem for now
    # Each quantum computer supports different gates, transpile needs to know what gates are available
    fake_backend = FakeBelemV2()  # TODO: Allow user to set backend as an arg so they can run on any quantum computer.

    # Transpile the circuits for optimized execution on the backend
    qc_smalls_h_t = []
    for qc in circuits_h:
        qc_smalls_h_t.append(transpile(qc, fake_backend, optimization_level=3))

    qc_smalls_v_t = []
    for qc in circuits_v:
        qc_smalls_v_t.append(transpile(qc, fake_backend, optimization_level=3))

    # combine all transpiled circuits into one list
    circ_list_t = qc_smalls_h_t.copy()
    circ_list_t.extend(qc_smalls_v_t)

    # Using token form arg[1] since load_account is not working
    service = QiskitRuntimeService(channel="ibm_quantum", token=TOKEN)
    # Make a new session with IBM
    # Set backend to "ibmq_qasm_simulator" for non-quantum results, for quantum results use "ibmq_belem" or other
    with Session(service=service, backend="ibmq_qasm_simulator") as session:
        sampler = Sampler(session=session)  # Make a Sampler to run the circuits
        # Executing the circuits on the backend
        job = sampler.run(circ_list_t, shots=8192)
        # job_monitor(job)  # job_monitor does not work
        print("\nJob queued, look to IBM website to get time updates.\nDO NOT CLOSE PROGRAM!!!")
        # Getting the resultant probability distribution after measurement
        result = job.result()  # Blocking until IBM returns with results

    #TODO: TURN ALL OF BELOW INTO 256X256

    ##NOTE::: COPIED CODE FROM 2X2 HARDWARE FOR TEMP PLACEHOLDER

    # Make a zeroed nparray of size chunk for each h and v chunk.
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

    # Map each chunk to its corresponding array.

    # Extract odd numbered states for each chunk. (maybe do it in the mapping above to save time?)
    edge_scan_small_h = np.array([counts_h[f'{2 * i + 1:03b}'] for i in range(2 ** data_qb)]).reshape(2, 2)
    edge_scan_small_v = np.array([counts_v[f'{2 * i + 1:03b}'] for i in range(2 ** data_qb)]).reshape(2, 2).T

    # Add together the H and V of each chunk.
    edge_detected_image_small = edge_scan_small_h + edge_scan_small_v


    # Stitch the chunks back into one image.

    # Plot edge detected image.
    plot_image(edge_detected_image_small, 'Full Edge Detected Image')

def main():
    print("Running 256x256 sim.")

    sim256x256()

if __name__ == "__main__":
    main()
