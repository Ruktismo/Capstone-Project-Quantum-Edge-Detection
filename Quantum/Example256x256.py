import numpy
from qiskit import *
from qiskit.compiler import transpile
from qiskit.providers.fake_provider.backends.guadalupe import fake_guadalupe
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
    plt.imshow(img, extent=[0, img.shape[0], img.shape[1], 0], cmap='viridis')
    # A blocking request to display the figure(s) loaded. Block ends when user closes figure(s)
    # Will show glitchy overlap if mutable figures are made before show is called
    plt.show()


# Convert the raw pixel values to probability amplitudes
# i.e. sum of all pixels is one
def amplitude_encode(img_data):
    # Calculate the RMS value
    rms = np.sqrt(np.sum(img_data ** 2))  # sum up all pixels to get total
    # if img has non-zero pixels
    if rms != 0:
        # Create normalized image
        image_norm = []
        for arr in img_data:
            for ele in arr:
                image_norm.append(ele / rms)  # divide pixel by total to get probability
    else:
        image_norm = [0.0] * img_data.size  # if rms is zero then chunk is empty

    # Return the normalized image as a numpy array
    ret = np.array(image_norm)
    return ret


def crop(image, hsize, vsize):
    h_chunks = image.shape[0] / hsize
    v_chunks = image.shape[1] / vsize
    # quick error check for hsplit and vsplit
    if (image.shape[0] % hsize != 0) or (image.shape[1] % vsize != 0):
        print("ERROR\nImage is not cleanly dividable by chunk size.")
        exit()
    # Split the image vertically then pump all vertical slices into hsplit to get square chunks
    nested_chunks = [np.hsplit(vs, h_chunks) for vs in np.vsplit(image, v_chunks)]
    # The split process leaves us with the chunks in a nested array, flatten to a list
    img_chunks = [item for sublist in nested_chunks for item in sublist]
    return img_chunks

# TODO this ia a copy of the state vector version. need to update to 16x16
def circuit_h(img, total_qb):
    # Create the circuit for horizontal scan
    #todo; this is 8x8, update to 16x16
    qc_h = QuantumCircuit(total_qb)
    qc_h.initialize(img, range(1, total_qb))
    qc_h.barrier()
    qc_h.h(0)
    qc_h.barrier()
    qc_h.x(0)
    qc_h.cx(0, 1)
    qc_h.ccx(0, 1, 2)
    qc_h.mcx([0, 1, 2], 3)
    qc_h.mcx([0, 1, 2, 3], 4)
    qc_h.mcx([0, 1, 2, 3, 4], 5)
    qc_h.mcx([0, 1, 2, 3, 4, 5], 6)
    qc_h.barrier()
    qc_h.h(0)
    qc_h.measure_all()

    return qc_h


def circuit_v(img, total_qb):
    # Create the circuit for vertical scan
    # todo; this is 8x8, update to 16x16
    qc_v = QuantumCircuit(total_qb)
    qc_v.initialize(img, range(1, total_qb))
    qc_v.barrier()
    qc_v.h(0)
    qc_v.barrier()
    qc_v.x(0)
    qc_v.cx(0, 1)
    qc_v.ccx(0, 1, 2)
    qc_v.mcx([0, 1, 2], 3)
    qc_v.mcx([0, 1, 2, 3], 4)
    qc_v.mcx([0, 1, 2, 3, 4], 5)
    qc_v.mcx([0, 1, 2, 3, 4, 5], 6)
    qc_v.barrier()
    qc_v.h(0)
    qc_v.measure_all()

    return qc_v

#####################################################
def sim256x256():
    style.use('bmh')  # This is setting the color scheme for the plots.
    pic = Image.open("./edge_detection_input.jpg")  # open image and crop to 256x256
    image_RGB = numpy.asarray(pic)

    # The image is in RGB, but we only need one BW
    # Convert the RBG component of the image to B&W image, as a numpy (uint8) array
    image = []
    for i in range(image_RGB.shape[0]):
        image.append([])
        for j in range(image_RGB.shape[1]):
            image[i].append(image_RGB[i][j][0] / 255.0)

    image = np.array(image)
    plot_image(image, 'Original Image')

    # Then crop the result into chunks
    croped_imgs = crop(image, H_SIZE, V_SIZE)

    # Get amplitude encoding of chunks for both H and V
    image_norms_h = []
    image_norms_v = []
    is_empty = [False] * 256  # holds a mapping of witch chunks we skipped over due to them being empty
    for chunk in range(len(croped_imgs)):
        if np.sum(croped_imgs[chunk]) == 0:
            # if chunk is empty then set is_empty and don't process it
            is_empty[chunk] = True
        else:
            # get amp encoding for H and V
            image_norms_h.append(amplitude_encode(croped_imgs[chunk]))
            image_norms_v.append(amplitude_encode(croped_imgs[chunk].T))

    # Get Vertical chunks: Transpose of Original image
    #todo: print the image_norms v and h here to see
    #print(image_norms_h)
    #print(image_norms_v)



    # Initialize some global variable for number of qubits
    data_qb = 8  # Set to ceil(log_2(image.CropSize)) hardcoded as 8 since image crop is 16x16
    anc_qb = 1  # This is the auxiliary qbit.
    total_qb = data_qb + anc_qb

    # make a circuit for each horizontal and vertical chunk
    circuits_h = []
    circuits_v = []
    for i in range(len(image_norms_h)):
        circuits_h.append(circuit_h(image_norms_h[i], total_qb))
        circuits_v.append(circuit_v(image_norms_v[i], total_qb))

    # combine all circuits into one list
    circuit_list = circuits_h
    circuit_list.extend(circuits_v)

    # Fake Backend for transpile to decompose circuit to. Locked to Belem for now
    # Each quantum computer supports different gates, transpile needs to know what gates are available
    fake_backend = fake_guadalupe()  # TODO: Allow user to set backend so they can run on any quantum computer.

    # Transpile the circuits for optimized execution on the backend
    # We made the circuits with high-level gates, need to decompose to basic gates so IBMQ hardware can understand
    for i in range(len(circuits_h)):
        qc_small_h_t = transpile(circuits_h[i], fake_backend, optimization_level=3)
        qc_small_v_t = transpile(circuits_v[i], fake_backend, optimization_level=3)

    # Combining both circuits into a list
    circ_list_t = [qc_small_h_t, qc_small_v_t]

    # Load the IBMQ account. It is not properly loading, but it is saved?
    # IBMQ.load_account()
    # Using token from arg[1] since load_account is not working
    service = QiskitRuntimeService(channel="ibm_quantum", token=TOKEN)
    # Make a new session with IBM
    # Set backend to "ibmq_qasm_simulator" for non-quantum results, for quantum results use "ibmq_belem" or other
    with Session(service=service, backend="ibmq_guadalupe") as session:
        sampler = Sampler(session=session)  # Make a Sampler to run the circuits
        # Executing the circuits on the backend
        job = sampler.run(circ_list_t, shots=8192)
        # job_monitor(job)  # job_monitor does not work
        print("\nJob queued, look to IBM website to get time updates.\nDO NOT CLOSE PROGRAM!!!")
        # Getting the resultant probability distribution after measurement
        result = job.result()  # Blocking until IBM returns with results




    counts_h = {f'{k:0{total_qb}b}': 0.0 for k in range(2 ** total_qb)}
    counts_v = {f'{k:0{total_qb}b}': 0.0 for k in range(2 ** total_qb)}

    # Make a zeroed nparray of size chunk for each h and v chunk.
    hArray = []
    vArray = []
    for zeroArrayH in image_norms_h:
        hArray.append(numpy.zeros((16, 16)))

    for zeroArrayV in image_norms_v:
        vArray.append(numpy.zeros((16, 16)))

    print(hArray)

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
