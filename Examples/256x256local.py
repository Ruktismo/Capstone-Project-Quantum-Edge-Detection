# Importing standard Qiskit libraries and configuring account
# Libs needed: qiskit, matplotlib, pylatexenc, qiskit-ibm-runtime
from qiskit import QuantumCircuit, execute, Aer

#standard libraries needed
import sys
import time
import math as m
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#TOKEN will be IBM quantum account API token
#be sure to copy from clipboard (NOT the keyboard shortcut)
try:
    CHUCK_SIZE = int(sys.argv[1])
    THREAD_COUNT = int(sys.argv[2])
except IndexError:
    print(f"ERROR: INCORRECT NUMBER OF ARGS\nExpected: [Chunk-Size,Thread-Count]\nGot: {sys.argv}")
    exit()
except ValueError:
    print("ERROR: CROP SIZE NOT OF TYPE INT\n\tUsing default of 16x16 and 1 thread")
    CHUCK_SIZE = 16

# Initialize global variable for number of qubits
data_qb = m.ceil(m.log2(CHUCK_SIZE**2)) # Set to ceil(log_2(image.CropSize)) hardcoded as 8 since image crop is 16x16
#h_size*v_size = dimension to plug in
#was originally data_qb = 8 for testing 16x16. equation will be for any size

anc_qb = 1  # This is the auxiliary qubit.
total_qb = data_qb + anc_qb  #total qubits

# Function for plotting the image using matplotlib
# parameters: image and title
def plot_image(img, title: str):
    plt.title(title)    #display the string title on the image
    plt.xticks([]) #remove ticks on x axis
    plt.yticks([]) #remove ticks on y axis

    #.imshow is built in function of plot library
    plt.imshow(img, extent=[0, img.shape[0], img.shape[1], 0], cmap='viridis', vmin=0.0, vmax=np.average(img))
    # A blocking request to display the figure(s) loaded. Block ends when user closes figure(s)
    # Will show glitchy overlap if mutable figures are made before show is called
    plt.show()


#Function to plot each chunk of the 256x256 image
def plot_chunks(chunks, shape_h, shape_v):
    #256//h_size, 256//v_size for hardcoding for testing. replace shape_h and shape_v with 256
    fig, axs = plt.subplots(shape_h//CHUCK_SIZE, shape_v//CHUCK_SIZE)
    index = 0
    #loop through each vertical and each horizontal
    for v in range(shape_v//CHUCK_SIZE):
        for h in range(shape_h//CHUCK_SIZE):
            # plot chunk
            axs[v,h].imshow(chunks[index], extent=[0, chunks[index].shape[0], chunks[index].shape[1], 0],
                            cmap='viridis')
            # remove all grid lines and tick marks
            axs[v, h].grid(False)
            axs[v, h].tick_params(axis='both', which='both', length=0)
            axs[v, h].set_xticks([])
            axs[v, h].set_yticks([])
            index += 1
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
        return None  # if rms is zero then chunk is empty

    # Return the normalized image as a numpy array
    ret = np.array(image_norm)
    return ret


#Function for building circuit; used one of glen's files for reference then adjusted
#will grow for any number of qubits needed (same for v)
def build_circuit(img):
    # Create the circuit for horizontal scan
    qc = QuantumCircuit(total_qb)
    qc.initialize(img, range(1, total_qb))
    qc.barrier()
    qc.h(0)
    qc.barrier()
    #TODO find a way to have the decrement gate NOT processes the right and bottom of the img.

    # Decrement gate - START
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
def process16x16(data: (np.array, int)):
    # Get horizontal and vertical amplitudes encoded pixel values
    image_norm_h = amplitude_encode(data[0])
    if image_norm_h is None:
        return None, data[1]
    image_norm_v = amplitude_encode(data[0].T)  # Image transpose for the vertical


    # Create the circuit for horizontal scan
    qc_h = build_circuit(image_norm_h)
    # Create the circuit for vertical scan
    qc_v = build_circuit(image_norm_v)

    # Combine both circuits into a single list
    circ_list = [qc_h, qc_v]

    #using QASM_SIMULATOR
    SHOTS = 2**10
    backend = Aer.get_backend('statevector_simulator')
    job = execute(circ_list, backend, shots=SHOTS)
    result = job.result().get_counts()


    # Make dic with keys that are binaries from 0 to 2^total_qb. with values of 0.0 to start.
    # This is done since IBM will not return Qbit configs that are 0. So we need to map the results to the full space
    # Formatted String: 0 (for padding zeros) {total_qb} (for bit-string size) b (to format from int to binary)
    counts_h = {f'{k:0{total_qb}b}': 0.0 for k in range(2 ** total_qb)}
    counts_v = {f'{k:0{total_qb}b}': 0.0 for k in range(2 ** total_qb)}

    # Transfer all known values form experiment results to dic
    for k, v in result[0].items():
        counts_h[k] = v / SHOTS
    for k, v in result[1].items():
        counts_v[k] = v / SHOTS


    # Extracting counts for odd-numbered states. i.e. data that we are interested in
    edge_scan_h = np.array([counts_h[f'{2 * i + 1:0{total_qb}b}'] for i in range(2 ** data_qb)]).reshape(CHUCK_SIZE, CHUCK_SIZE)  #horizontal
    edge_scan_v = np.array([counts_v[f'{2 * i + 1:0{total_qb}b}'] for i in range(2 ** data_qb)]).reshape(CHUCK_SIZE, CHUCK_SIZE).T  #transpose for vertical

    # Combine H and V scans to get full edge detection image
    edge_scan_sim = edge_scan_h + edge_scan_v

    return edge_scan_sim, data[1]


#Function to crop the image
def crop(image, c_size):
    h_chunks = image.shape[0] / c_size
    v_chunks = image.shape[1] / c_size

    # quick error check for hsplit and vsplit
    if (image.shape[0] % c_size != 0) or (image.shape[1] % c_size != 0):
        print("ERROR\nImage is not cleanly dividable by chunk size.")
        exit()

    # Split the image vertically then pump all vertical slices into hsplit to get square chunks
    nested_chunks = [np.hsplit(vs, h_chunks) for vs in np.vsplit(image, v_chunks)]

    # The split process leaves us with the chunks in a nested array, flatten to a list
    img_chunks = [item for sublist in nested_chunks for item in sublist]

    return img_chunks


def sim256x256():
    pic = Image.open("./256 Test Images/edge_detection_input.jpg")  # open image and crop to 256x256
    image_RGB = np.asarray(pic)

    # The image is in RGB, but we only need one BW
    # Convert the RBG component of the image to B&W image, as a numpy (uint8) array
    image = []
    for i in range(image_RGB.shape[0]):
        image.append([])
        for j in range(image_RGB.shape[1]):
            image[i].append(image_RGB[i][j][0] / 255.0)  # TODO clamp to 0 or 1

    image = np.array(image)
    # plot_image(image, 'Original Image')

    # Then crop the result into chunks
    croped_imgs = crop(image, CHUCK_SIZE)
    #put this back later. will give size. displays image in chunk form. dont want to do every time for now.
    #plot_chunks(croped_imgs, image.shape[0], image.shape[1])
# Processing Start
    is_empty = [None] * len(croped_imgs)
    edge_detected_image = [None] * len(croped_imgs)
    tic = time.perf_counter()
    with Pool(processes=THREAD_COUNT) as pool:  #future: change processes number? leave 10 for now.
        results = pool.imap_unordered(process16x16, [(croped_imgs[N],N) for N in range(len(croped_imgs))])
        for r in results:
            print(f"Chunk {r[1]} processed")
            if r[0] is None:
                is_empty[r[1]] = True
            else:
                is_empty[r[1]] = False
                edge_detected_image[r[1]] = r[0]
    edge_detected_image = [c for c in edge_detected_image if isinstance(c, np.ndarray)]
    print(f"Nones:{len([e for e in is_empty if e])}")  # Count how many where not processed
    # calculate ~time it will take to run
    tok = time.perf_counter()
    t = tok - tic
    print(f"Total Compile/Run Time: {t:0.4f} seconds")
# Processing End

    # Stitch the chunks back into one image.
    # first make empty image
    ed_image_arr = np.zeros((256, 256))
    ed_image_chunks = []
    res = 0  # index of current result

    # loop for upper left box for each chunk
    for i in range(len(is_empty)):
        # if there was data that was processed
        if not is_empty[i]:
            # paste it in to the image
            #ULBox = (i//16*16, (i*16)%256) #was to be hard coded for testing
            ULBox = (i // (image.shape[0] // CHUCK_SIZE) * CHUCK_SIZE,
                     (i * CHUCK_SIZE) % image.shape[0])  # find upper left cords of chunk based off of chunk index

            ed_image_arr[ULBox[0]+1:ULBox[0] + CHUCK_SIZE-1, ULBox[1]+1:ULBox[1] + CHUCK_SIZE-1] += \
                edge_detected_image[res][1:CHUCK_SIZE-1,1:CHUCK_SIZE-1]

            #ed_image_arr[ULBox[0]:ULBox[0]+CHUCK_SIZE, ULBox[1]:ULBox[1]+CHUCK_SIZE] += edge_detected_image[res]

            #ed_image_chunks.append(edge_detected_image[res])
            res += 1  # move res to next result
        # If not then leave as default black
        else:
            pass#ed_image_chunks.append(np.zeros((CHUCK_SIZE, CHUCK_SIZE)))  # 16x16, (16,16) for hard coded testing

    # don't know if this works, if it does then remove try/catch
    # plot_chunks(ed_image_chunks, image.shape[0], image.shape[1])

    # Plot edge detected image.
    plot_image(ed_image_arr, 'Full Edge Detected Image')


# MAIN
def main():
    print('*************************************************')
    print('running local 256x256 simulation...')
    print('*************************************************')
    tic = time.perf_counter()
    sim256x256()
    # calculate ~time it will take to run
    tok = time.perf_counter()
    t = tok - tic
    print(f"Total Start/End Time: {t:0.4f} seconds")

if __name__ == "__main__":
    main()