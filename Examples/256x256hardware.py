"""
Keeping this file for reference. It may not be needed any more.
TODO: see if it's safe to delete
TODO: if keeping update multiprocessing.Pool to concurrent.futures.ProcessPoolExecutor
"""

#Qiskit libraries needed
from qiskit import *
from qiskit.compiler import transpile
from qiskit.providers.fake_provider.backends.guadalupe.fake_guadalupe import FakeGuadalupeV2
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler

#import standard libraries
import sys
import time
import random as ran
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from PIL import Image
from matplotlib import style
import math as Math

# :)

#TOKEN will be IBM quantum account API token
#be sure to copy from clipboard (NOT the keyboard shortcut)
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

# Initialize global variable for number of qubits
data_qb = Math.ceil(Math.log2(H_SIZE*V_SIZE)) # Set to ceil(log_2(image.CropSize)) hardcoded as 8 since image crop is 16x16
#h_size*v_size = dimension to plug in
#was originally data_qb = 8 for testing 16x16. equation will be for any size

anc_qb = 1  # This is the auxiliary qubit.
total_qb = data_qb + anc_qb  #total qubits

# Function for plotting the image using matplotlib
def plot_image(img, title: str):
    plt.title(title)
    plt.imshow(img, extent=[0, img.shape[0], img.shape[1], 0], cmap='viridis')
    plt.xticks([])
    plt.yticks([])
    # A blocking request to display the figure(s) loaded. Block ends when user closes figure(s)
    # Will show glitchy overlap if mutable figures are made before show is called
    plt.show()


#Function to plot each chunk of the 256x256 image
def plot_chunks(chunks, shape_h, shape_v):
    #256//h_size, 256//v_size for hardcoding for testing. replace shape_h and shape_v with 256
    fig, axs = plt.subplots(shape_h//H_SIZE, shape_v//V_SIZE)
    index = 0
    #loop through each vertical and each horizontal
    for v in range(shape_v//V_SIZE):
        for h in range(shape_h//H_SIZE):
            # plot chunk
            axs[v,h].imshow(chunks[index], extent=[0, chunks[index].shape[0], chunks[index].shape[1], 0],
                            cmap='viridis', vmin=0.0, vmax=1.0)
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
        image_norm = [0.0] * img_data.size  # if rms is zero then chunk is empty

    # Return the normalized image as a numpy array
    ret = np.array(image_norm)
    return ret

#Function to crop the image
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

#Function for horizontal circuit; used one of glen's files for reference then adjusted
#will grow for any number of qubits needed (same for v)
def circuit_h(img):
    # Create the circuit for horizontal scan
    qc_h = QuantumCircuit(total_qb)
    qc_h.initialize(img, range(1, total_qb)) #todo? maybe look at this for efficiency too? (lower priority)
    qc_h.barrier()
    qc_h.h(0)
    qc_h.barrier()
    #TODO: MAIN FOCUS: DECREMENT GATE; EFFICIENCY!! <---
    # Decrement gate - START
    qc_h.x(0)
    qc_h.cx(0, 1)
    qc_h.ccx(0, 1, 2) #todo? take look at ccx (reducing complexity)
    for c in range(3,total_qb):
        qc_h.mcx([b for b in range(c)], c)
    # Decrement gate - END
    qc_h.barrier()
    qc_h.h(0)
    qc_h.measure_all()

    return qc_h

#Function for vertical circuit; used one of glen's files for reference then adjusted
def circuit_v(img):
    # Create the circuit for vertical scan
    qc_v = QuantumCircuit(total_qb)
    qc_v.initialize(img, range(1, total_qb))
    qc_v.barrier()
    qc_v.h(0)
    qc_v.barrier()
    # Decrement gate - START
    qc_v.x(0)
    qc_v.cx(0, 1)
    qc_v.ccx(0, 1, 2)
    for c in range(3,total_qb):
        qc_v.mcx([b for b in range(c)], c)
    # Decrement gate - END
    qc_v.barrier()
    qc_v.h(0)
    qc_v.measure_all()

    return qc_v

#Pre-processing for each chunk
def pre_processing(data: (np.array, int)):
    # Get amplitude encoding of chunks for both H and V
    if np.sum(data[0]) == 0:
        # if chunk is empty then don't need to process it
        return None, None, data[1]
    else:
        # get amp encoding for H and V
        image_norms_h = amplitude_encode(data[0])
        image_norms_v = amplitude_encode(data[0].T)

    # make a circuit for each horizontal and vertical chunk
    circuits_h = circuit_h(image_norms_h)
    circuits_v = circuit_v(image_norms_v)

    # Fake Backend for transpile to decompose circuit to. Locked to Guadalupe for now
    # Each quantum computer supports different gates, transpile needs to know what gates are available
    fake_backend = FakeGuadalupeV2()  # TODO: Allow user to set backend so they can run on any quantum computer.

    done = False
    while not done:
        # Transpile the circuits for optimized execution on the backend
        # We made the circuits with high-level gates, need to decompose to basic gates so IBMQ hardware can understand
        try:
            #work around is using the seed for eigenvalue issues. gives random seed initially, but if you give a seed, can produce new circuit
            seed = ran.randint(1,2**32)
            qc_small_h_t = transpile(circuits_h, fake_backend, optimization_level=3, seed_transpiler=seed)
            qc_small_v_t = transpile(circuits_v, fake_backend, optimization_level=3, seed_transpiler=seed)
            done = True
        except np.linalg.LinAlgError:
            # numpy.linalg.LinAlgError: eig algorithm (geev) did not converge (only eigenvalues with order >= 4 have converged)
            print(f"Chunk {data[1]}:\tEigen Value Error: re-transpiling")
            pass

    # Combining both circuits into a list
    circ_list_t_hv = (qc_small_h_t, qc_small_v_t, data[1])
    return circ_list_t_hv


#####################################################
def sim256x256():
    style.use('bmh')  # This is setting the color scheme for the plots.
    pic = Image.open("./256 Test Images/edge_detection_input.jpg")  # open image and crop to 256x256
    image_RGB = np.asarray(pic)

    # The image is in RGB, but we only need one BW
    # Convert the RBG component of the image to B&W image, as a numpy (uint8) array
    image = []
    for i in range(image_RGB.shape[0]):
        image.append([])
        for j in range(image_RGB.shape[1]):
            image[i].append(image_RGB[i][j][0] / 255.0)

    image = np.array(image)
    #plot_image(image, 'Original Image')

    # Then crop the result into chunks
    croped_imgs = crop(image, H_SIZE, V_SIZE)
    #put this back later. will give size. displays image in chunk form. dont want to do every time for now.
    #plot_chunks(croped_imgs, image.shape[0], image.shape[1])


#Pre-Processing Start
    is_empty = [None] * len(croped_imgs)
    circ_list_t_h = [None] * len(croped_imgs)
    circ_list_t_v = [None] * len(croped_imgs)
    # Locking to max of 10 processes since anymore and we start to fight with the rest of the computer for RAM and CPU
    # If running on a computer with more than 8 cores and 16Gb of RAM then you can increase
    tic = time.perf_counter()
    with Pool(processes=10) as pool:  #future: change processes number? leave 10 for now.
        results = pool.imap_unordered(pre_processing, [(croped_imgs[N],N) for N in range(len(croped_imgs))])
        for r in results:
            print(f"Chunk {r[2]} processed")
            if r[0] is None:
                is_empty[r[2]] = True
            else:
                is_empty[r[2]] = False
                circ_list_t_h[r[2]] = r[0]
                circ_list_t_v[r[2]] = r[1]

    # combine all circuits into one ordered list all horizontal first then vertical
    #circ_list_t = circ_list_t_h
    #circ_list_t.extend(circ_list_t_v)
    circ_list_t_h = [c for c in circ_list_t_h if isinstance(c, QuantumCircuit)] # Remove all Nones form the list
    circ_list_t_v = [c for c in circ_list_t_v if isinstance(c, QuantumCircuit)]  # Remove all Nones form the list
    print(f"Circ h:{len(circ_list_t_h)}")
    print(f"Circ v:{len(circ_list_t_v)}")
    print(f"Nones:{len([e for e in is_empty if e])}") # Count how many where not processed
    print(f"Nones List: {is_empty}")

    #calculate ~time it will take to run
    tok = time.perf_counter()
    t = tok - tic
    print(f"Total Time: {t:0.4f} seconds")

#Pre-Processing End

    # Load the IBMQ account. It is not properly loading, but it is saved?
    # IBMQ.load_account()
    # Using token from arg[1] since load_account is not working
    service = QiskitRuntimeService(channel="ibm_quantum", token=TOKEN)
    # Make a new session with IBM
    # Set backend to "ibmq_qasm_simulator" for non-quantum results, for quantum results use "ibmq_belem" or other
    with Session(service=service, backend="ibmq_qasm_simulator") as session:
            #todo; adjust ibmq
        sampler_h = Sampler(session=session)  # Make a Sampler to run the circuits
        # Executing the circuits on the backend
        job_h = sampler_h.run(circ_list_t_h, shots=8192)
        sampler_v = Sampler(session=session)  # Make a Sampler to run the circuits

        # Executing the circuits on the backend
        job_v = sampler_v.run(circ_list_t_v, shots=8192)

        print("\nJob queued, look to IBM website to get time updates.\nDO NOT CLOSE PROGRAM!!!")

        # Getting the resultant probability distribution after measurement
        result_h = job_h.result()  # Blocking until IBM returns with results
        result_v = job_v.result()

    edge_detected_image = []
    # for each pair of H and V circuits we have
    for i in range(len(circ_list_t_h)):
        counts_h = {f'{k:0{total_qb}b}': 0.0 for k in range(2 ** total_qb)}  # create binaries
        counts_v = {f'{k:0{total_qb}b}': 0.0 for k in range(2 ** total_qb)}

        # Transfer all known values form experiment results to dic
        for k, v in result_h[i].items():
            counts_h[format(k, f"0{total_qb}b")] = v * 255  # And convert from probability to color int between 0-255

        for k, v in result_v[i].items():
            counts_v[format(k, f"0{total_qb}b")] = v * 255

        # Extract odd numbered states for each chunk. (maybe do it in the mapping above to save time?)
        edge_scan_h = np.array(
            [counts_h[f'{2 * i + 1:0{total_qb}b}'] for i in range(2 ** data_qb)]
        ).reshape(H_SIZE, V_SIZE)

        edge_scan_v = np.array(
            [counts_v[f'{2 * i + 1:0{total_qb}b}'] for i in range(2 ** data_qb)]
        ).reshape(H_SIZE, V_SIZE).T
        # combine h and v into one chunk
        edge_detected_image.append(edge_scan_h + edge_scan_v)

    # Stitch the chunks back into one image.
    # first make empty image
    ed_image = Image.new('L', (image.shape[0], image.shape[1]))  # L mode is int between 0-255;; #(256,256 hard coded for initial testing)
    ed_image_chunks = []
    res = 0 # index of current result

    #loop for upper left box for each chunk
    for i in range(len(is_empty)):
        # if there was data that was processed
        if not is_empty[i]:
            # paste it in to the image
            #ULBox = (i//16*16, (i*16)%256) was to be hard coded for testing
            ULBox = (i//H_SIZE*H_SIZE, (i*V_SIZE) % image.shape[0]) # find upper left cords of chunk based off of chunk index
            ed_image.paste(Image.fromarray(edge_detected_image[res], mode='L'), box=ULBox)  # paste chunk
            ed_image_chunks.append(edge_detected_image[res])
            res += 1 # move res to next result
        # If not then leave as default black
        else:
            ed_image_chunks.append(np.zeros((H_SIZE, V_SIZE))) #16x16, (16,16) for hard coded testing

    # don't know if this works, if it does then remove try/catch
    plot_chunks(ed_image_chunks, image.shape[0], image.shape[1])

    # Plot edge detected image.
    plot_image(np.array(ed_image), 'Full Edge Detected Image')

# MAIN
def main():
    print('*************************************************')
    print("Running 256x256 sim.")
    print('*************************************************')

    sim256x256()


if __name__ == "__main__":
    main()
