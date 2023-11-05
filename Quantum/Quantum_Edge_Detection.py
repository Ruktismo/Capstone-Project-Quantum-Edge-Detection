# Importing standard Qiskit libraries and configuring account
# Libs needed:
#   matplotlib, pillow, qiskit, pylatexenc, qiskit-ibm-runtime, qiskit-aer

from qiskit import QuantumCircuit, execute, Aer
import logging
import os
import sys
import time
from PIL import Image
import math as m
# from multiprocessing import Pool  Will work for windows but not linux. Instead use
from concurrent.futures import ProcessPoolExecutor as Pool
import numpy as np
import configparser
import argparse

# get logger
log = logging.getLogger("Quantum_Edge_Detection")

# Create a configparser object
config = configparser.ConfigParser()
# Read an existing configuration file
configFilePath = os.path.dirname(__file__)+"/../Config.ini"  # get the path to Config.ini relative to this file
config.read_file(open(configFilePath))

class QED:
    # Do QED with args from config file if none are provided
    def __init__(self, chunkSize=int(config['QED']['CHUNK_SIZE']),
                 shots=int(config['QED']['SHOTS']),
                 threadCount=int(config['QED']['QED_THREAD_COUNT'])):
        log.debug("setting up QED")
        self.CHUNK_SIZE = chunkSize
        self.THREAD_COUNT = threadCount
        self.SHOTS = shots  # Number of runs to do. More runs gets better quality. But also more time
        self.data_qb = m.ceil(m.log2(self.CHUNK_SIZE ** 2))  # Set to ceil(log_2(image.CropSize))
        self.anc_qb = 1  # This is the auxiliary qubit.
        self.total_qb = self.data_qb + self.anc_qb  # total qubits

    # Convert the raw pixel values to probability amplitudes
    # i.e. sum of all pixels is one
    def amplitude_encode(self, img_data):
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

    # Function for building circuit; used one of glen's files for reference then adjusted
    # will grow for any number of qubits needed (same circuit for horizontal/vertical)
    def build_qed_circuit(self, img):
        # Create the circuit for horizontal scan
        qc = QuantumCircuit(self.total_qb)
        qc.initialize(img, range(1, self.total_qb))
        qc.barrier()
        qc.h(0)
        qc.barrier()

        # Decrement gate - START
        # TODO find a way to have the decrement gate NOT processes the borders of the img.
        qc.x(0)
        qc.cx(0, 1)
        qc.ccx(0, 1, 2)
        for c in range(3, self.total_qb):
            qc.mcx([b for b in range(c)], c)
        # Decrement gate - END
        qc.barrier()
        qc.h(0)
        qc.measure_all()

        return qc

    # 16x16 image simulation
    # np.array: Image to be processed
    # int: index of what chunk this image is
    def process16x16(self, data: (np.array, int)):
        # Get horizontal and vertical amplitudes encoded pixel values
        image_norm_h = self.amplitude_encode(data[0])
        if image_norm_h is None:
            return None, data[1]  # if chunk is all zeros then there is nothing to process
        image_norm_v = self.amplitude_encode(data[0].T)
        # Image transpose for the vertical, so vertical is treated like horizontal

        # Create the circuit for horizontal scan
        qc_h = self.build_qed_circuit(image_norm_h)
        # Create the circuit for vertical scan
        qc_v = self.build_qed_circuit(image_norm_v)

        # Combine both circuits into a single list
        circ_list = [qc_h, qc_v]

        # using QASM_SIMULATOR

        backend = Aer.get_backend('statevector_simulator')
        job = execute(circ_list, backend, shots=self.SHOTS)  # Run job
        result = job.result().get_counts()  # Get results

        # Make dic with keys that are binaries from 0 to 2^total_qb. with values of 0.0 to start.
        # This is done since IBM will not return Qbit strings that are 0.
        # So we need to map the results to the full image space.
        # Formatted String: 0 (for padding zeros) {total_qb} (for bit-string size) b (to format from int to binary)
        counts_h = {f'{k:0{self.total_qb}b}': 0.0 for k in range(2 ** self.total_qb)}
        counts_v = {f'{k:0{self.total_qb}b}': 0.0 for k in range(2 ** self.total_qb)}

        # Transfer all known values form experiment results to dic.
        # Normalising to the number of shots to keep results between [0,1].
        for k, v in result[0].items():
            counts_h[k] = v / self.SHOTS
        for k, v in result[1].items():
            counts_v[k] = v / self.SHOTS

        # Extracting counts for odd-numbered states. i.e. data that we are interested in
        # And reshaping back into 2D image.
        edge_scan_h = np.array([counts_h[f'{2 * i + 1:0{self.total_qb}b}'] for i in range(2 ** self.data_qb)]) \
            .reshape(self.CHUNK_SIZE, self.CHUNK_SIZE)  # horizontal
        edge_scan_v = np.array([counts_v[f'{2 * i + 1:0{self.total_qb}b}'] for i in range(2 ** self.data_qb)]) \
            .reshape(self.CHUNK_SIZE, self.CHUNK_SIZE).T  # transpose vertical back to right side up

        # Combine H and V scans to get full edge detection image
        edge_scan_sim = edge_scan_h + edge_scan_v

        return edge_scan_sim, data[1]

    #Function to crop the image
    def crop(self, image):
        # quick error check for split functions.
        if (image.shape[0] % self.CHUNK_SIZE != 0) or (image.shape[1] % self.CHUNK_SIZE != 0):
            log.error("ERROR\n\tImage is not cleanly dividable by chunk size.")
            exit(-1)

        v_chunks = image.shape[0] / self.CHUNK_SIZE
        h_chunks = image.shape[1] / self.CHUNK_SIZE

        # Split the image vertically then pump all vertical slices into hsplit to get square chunks
        nested_chunks = [np.hsplit(vs, h_chunks) for vs in np.vsplit(image, v_chunks)]

        # The split process leaves us with the chunks in a nested array, flatten to a list
        img_chunks = [item for sublist in nested_chunks for item in sublist]

        return img_chunks

    def run_QED(self, pic):
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
                        "\tExpected types: PIL.Image.Image, File Path to image, NumPy Array\n"
                        f"\tGot: {pic}\t{type(pic)}")
            return None

        # The image is in RGB, but we only need B&W
        # Convert the RBG component of the image to B&W image, as a numpy (uint8) array
        image = []
        for i in range(image_RGB.shape[0]):
            image.append([])
            for j in range(image_RGB.shape[1]):
                image[i].append(image_RGB[i][j][0] / 255.0)

        image = np.array(image)

        # Then crop the result into chunks
        croped_imgs = self.crop(image)
    # Processing Start
        log.debug("Starting up worker pool")
        is_empty = [None] * len(croped_imgs)
        edge_detected_image = [None] * len(croped_imgs)
        tic = time.perf_counter()
        with Pool(max_workers=self.THREAD_COUNT) as pool:
            results = pool.map(self.process16x16, [(croped_imgs[N],N) for N in range(len(croped_imgs))])
            for r in results:
                log.debug(f"Chunk {r[1]} processed")
                if r[0] is None:
                    is_empty[r[1]] = True
                else:
                    is_empty[r[1]] = False
                    edge_detected_image[r[1]] = r[0]
        log.debug("All workers finished")
        # not all chunks may be processed, so shorten list to only be processed chunks
        edge_detected_image = [c for c in edge_detected_image if isinstance(c, np.ndarray)]
        nonesCount = len([e for e in is_empty if e])
        if nonesCount == len(is_empty):
            log.error("Image is entirely empty")
            exit(-7)
        log.debug(f"Nones count:{nonesCount}")  # Count how many where not processed
        # calculate ~time it took to run
        tok = time.perf_counter()
        t = tok - tic
        log.debug(f"Total Compile/Run Time: {t:0.4f} seconds")
    # Processing End

        # Stitch the chunks back into one image.
        # first make empty image
        ed_image_arr = np.zeros((image_RGB.shape[0], image_RGB.shape[1]))
        res = 0  # index of current result

        # Iterator for getting the indexes for pasting
        # defined within QED, so it can use image.shape and chunk size without issue
        def ULBox_iterator():
            for upper_left_x_cord in range(0, image.shape[0], self.CHUNK_SIZE):
                for upper_left_y_cord in range(0, image.shape[1], self.CHUNK_SIZE):
                    yield upper_left_x_cord, upper_left_y_cord
        box_gen = ULBox_iterator()  # create a generator obj to give the cords

        # loop for upper left box for each chunk
        for i in range(len(is_empty)):
            # if there was data that was processed
            if not is_empty[i]:
                # paste it in to the image
                # find upper left cords of chunk based off of chunk index
                ULBox = next(box_gen)  # ask the generator for the next set of cords
                # paste ed_results into ed_image, cutting out edge pixels
                ed_image_arr[ULBox[0]+1:ULBox[0] + self.CHUNK_SIZE - 1, ULBox[1] + 1:ULBox[1] + self.CHUNK_SIZE - 1] += \
                    edge_detected_image[res][1:self.CHUNK_SIZE - 1, 1:self.CHUNK_SIZE - 1]

                res += 1  # move res to next result
            # If not then leave as default black
            else:
                pass
        # scale image up so it can be seen as pixels
        ed_image_arr *= 1000000
        # return edge detected image.
        return ed_image_arr


def main():
    # if we are main we have to set up the stream handler
    formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(name)s : %(funcName)s : %(message)s")
    # Stream handler to output to stdout
    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setLevel(logging.INFO)  # handlers can set their logging independently or take the parent.
    log_stream_handler.setFormatter(formatter)
    # add handlers to log
    log.addHandler(log_stream_handler)

    parser = argparse.ArgumentParser(description="Run quantum edge detection (QED) on a photo")
    parser.add_argument('-f','--file',
                        help="File to be processed",
                        required=True)
    parser.add_argument('-t', '--threads', type=int,
                        default=int(config['QED']['QED_THREAD_COUNT']),
                        help="Max threads that QED can use")
    parser.add_argument('-c', '--chunk', type=int,
                        default=int(config['QED']['CHUNK_SIZE']),
                        help="Square size to chop up image with. Must be dividable by the with and height of the image.")
    parser.add_argument('-s', '--shots', type=int,
                        default=int(config['QED']['SHOTS']),
                        help="Number of shots to run in the simulator. More shots will take longer to run.")
    parser.add_argument('-o', '--output',
                        default=None,
                        help="Filename/location to save the result")
    parser.add_argument('-S', '--Scale', type=int,
                        default=1000000,
                        help="Scaling factor to make the image more visible\n\tIf set too low image may be all black")

    args = parser.parse_args()
    if args.output is None:
        args.output = "QED-" + args.file # default file name

    qed = QED(args.chunk, args.shots, args.threads)
    log.info(f"Running QED on: {args.file}")
    processed = qed.run_QED(args.file)
    processed *= 1000000
    log.info(f"QED finished, saving as: {args.file}")
    img = Image.fromarray(processed).convert("L")
    img.save(args.output)


if __name__ == "__main__":
    main()
    #default_QED = QED()
    #default_QED.run_QED("./LEFT.jpg")
