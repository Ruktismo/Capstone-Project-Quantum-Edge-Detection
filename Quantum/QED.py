# Importing standard Qiskit libraries and configuring account
# Libs needed: qiskit, matplotlib, pylatexenc, qiskit-ibm-runtime
import os.path

from qiskit import QuantumCircuit, execute, Aer

# Standard libraries used
import logging
import time
from PIL import Image
import math as m
from multiprocessing import Pool
import numpy as np
import configparser

# get logger, use __name__ to prevent conflict
log = logging.getLogger(__name__)
log.info("setting up QED")

# Create a configparser object
config = configparser.ConfigParser()
# Read an existing configuration file
config.read_file(open("./../Config.ini"))
# from config group QED get var CHUNK_SIZE
CHUNK_SIZE = int(config['QED']['CHUNK_SIZE'])  # should be int, needs casting since all config are read as str
THREAD_COUNT = int(config['QED']['QED_THREAD_COUNT'])

# TODO Error check for config

# Initialize global variable for number of qubits
data_qb = m.ceil(m.log2(CHUNK_SIZE ** 2)) # Set to ceil(log_2(image.CropSize))
anc_qb = 1  # This is the auxiliary qubit.
total_qb = data_qb + anc_qb  #total qubits

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
        return None  # if rms is zero then chunk is empty

    # Return the normalized image as a numpy array
    ret = np.array(image_norm)
    return ret


#Function for building circuit; used one of glen's files for reference then adjusted
#will grow for any number of qubits needed (same circuit for horizontal/vertical)
def build_qed_circuit(img):
    # Create the circuit for horizontal scan
    qc = QuantumCircuit(total_qb)
    qc.initialize(img, range(1, total_qb))
    qc.barrier()
    qc.h(0)
    qc.barrier()

    # Decrement gate - START
    # TODO find a way to have the decrement gate NOT processes the borders of the img.
    qc.x(0)
    qc.cx(0, 1)
    qc.ccx(0, 1, 2)
    for c in range(3,total_qb):
        qc.mcx([b for b in range(c)], c)
    # Decrement gate - END
    qc.barrier()
    qc.h(0)
    qc.measure_all()

    return qc


# 16x16 image simulation
# np.array: Image to be processed
# int: index of what chunk this image is
def process16x16(data: (np.array, int)):
    # Get horizontal and vertical amplitudes encoded pixel values
    image_norm_h = amplitude_encode(data[0])
    if image_norm_h is None:
        return None, data[1]  # if chunk is all zeros then there is nothing to process
    image_norm_v = amplitude_encode(data[0].T)  # Image transpose for the vertical, so vertical is treated like horizontal

    # Create the circuit for horizontal scan
    qc_h = build_qed_circuit(image_norm_h)
    # Create the circuit for vertical scan
    qc_v = build_qed_circuit(image_norm_v)

    # Combine both circuits into a single list
    circ_list = [qc_h, qc_v]

    #using QASM_SIMULATOR
    SHOTS = 2**10  # Number of runs to do. More runs gets better quality. But also more time
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circ_list, backend, shots=SHOTS)  # Run job
    result = job.result().get_counts()  # Get results

    # Make dic with keys that are binaries from 0 to 2^total_qb. with values of 0.0 to start.
    # This is done since IBM will not return Qbit strings that are 0.
    # So we need to map the results to the full image space.
    # Formatted String: 0 (for padding zeros) {total_qb} (for bit-string size) b (to format from int to binary)
    counts_h = {f'{k:0{total_qb}b}': 0.0 for k in range(2 ** total_qb)}
    counts_v = {f'{k:0{total_qb}b}': 0.0 for k in range(2 ** total_qb)}

    # Transfer all known values form experiment results to dic.
    # Normalising to the number of shots to keep results between [0,1].
    for k, v in result[0].items():
        counts_h[k] = v / SHOTS
    for k, v in result[1].items():
        counts_v[k] = v / SHOTS

    # Extracting counts for odd-numbered states. i.e. data that we are interested in
    # And reshaping back into 2D image.
    edge_scan_h = np.array([counts_h[f'{2 * i + 1:0{total_qb}b}'] for i in range(2 ** data_qb)])\
                    .reshape(CHUNK_SIZE, CHUNK_SIZE)  #horizontal
    edge_scan_v = np.array([counts_v[f'{2 * i + 1:0{total_qb}b}'] for i in range(2 ** data_qb)])\
                    .reshape(CHUNK_SIZE, CHUNK_SIZE).T  #transpose vertical back to right side up

    # Combine H and V scans to get full edge detection image
    edge_scan_sim = edge_scan_h + edge_scan_v

    return edge_scan_sim, data[1]


#Function to crop the image
def crop(image, c_size):
    # quick error check for split functions.
    if (image.shape[0] % c_size != 0) or (image.shape[1] % c_size != 0):
        log.error("ERROR\n\tImage is not cleanly dividable by chunk size.")
        exit(-1)

    h_chunks = image.shape[0] / c_size
    v_chunks = image.shape[1] / c_size

    # Split the image vertically then pump all vertical slices into hsplit to get square chunks
    nested_chunks = [np.hsplit(vs, h_chunks) for vs in np.vsplit(image, v_chunks)]

    # The split process leaves us with the chunks in a nested array, flatten to a list
    img_chunks = [item for sublist in nested_chunks for item in sublist]

    return img_chunks


def QED(pic):
    # attempt to open image and/or convert to np.array
    if isinstance(pic, Image.Image):
        image_RGB = np.asarray(pic)
    elif os.path.isfile(pic):
        img = Image.open(pic)
        image_RGB = np.asarray(img)
    elif isinstance(pic, np.ndarray):
        image_RGB = pic  # nothing to do just checking its one of the valid types
    else:
        log.warning("Invalid type given to QED\n"
                    "Expected types: PIL.Image.Image, File Path to image, NumPy Array\n"
                    f"Got: {type(pic)}")
        return None

    # The image is in RGB, but we only need B&W
    # Convert the RBG component of the image to B&W image, as a numpy (uint8) array
    image = []
    for i in range(image_RGB.shape[0]):
        image.append([])
        for j in range(image_RGB.shape[1]):
            image[i].append(image_RGB[i][j][0] / 255.0)  # TODO? clamp to 0 or 1

    image = np.array(image)

    # Then crop the result into chunks
    croped_imgs = crop(image, CHUNK_SIZE)
# Processing Start
    is_empty = [None] * len(croped_imgs)
    edge_detected_image = [None] * len(croped_imgs)
    tic = time.perf_counter()
    with Pool(processes=THREAD_COUNT) as pool:
        results = pool.imap_unordered(process16x16, [(croped_imgs[N],N) for N in range(len(croped_imgs))])
        for r in results:
            log.debug(f"Chunk {r[1]} processed")
            if r[0] is None:
                is_empty[r[1]] = True
            else:
                is_empty[r[1]] = False
                edge_detected_image[r[1]] = r[0]
    # not all chunks may be processed, so shorten list to only be processed chunks
    edge_detected_image = [c for c in edge_detected_image if isinstance(c, np.ndarray)]
    log.debug(f"Nones count:{len([e for e in is_empty if e])}")  # Count how many where not processed
    # calculate ~time it will take to run
    tok = time.perf_counter()
    t = tok - tic
    log.debug(f"Total Compile/Run Time: {t:0.4f} seconds")
# Processing End

    # Stitch the chunks back into one image.
    # first make empty image
    ed_image_arr = np.zeros((image_RGB.shape[0], image_RGB.shape[1]))
    res = 0  # index of current result

    # loop for upper left box for each chunk
    for i in range(len(is_empty)):
        # if there was data that was processed
        if not is_empty[i]:
            # paste it in to the image

            # find upper left cords of chunk based off of chunk index
            ULBox = (i // (image.shape[0] // CHUNK_SIZE) * CHUNK_SIZE,
                     (i * CHUNK_SIZE) % image.shape[0])
            # paste ed_results into ed_image, cutting out edge pixels
            ed_image_arr[ULBox[0]+1:ULBox[0] + CHUNK_SIZE - 1, ULBox[1] + 1:ULBox[1] + CHUNK_SIZE - 1] += \
                edge_detected_image[res][1:CHUNK_SIZE - 1, 1:CHUNK_SIZE - 1]

            res += 1  # move res to next result
        # If not then leave as default black
        else:
            pass

    # return edge detected image.
    return ed_image_arr
