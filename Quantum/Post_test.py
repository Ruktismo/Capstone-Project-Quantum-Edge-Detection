import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import style

from TestReasults import is_empty
from TestReasults import result_h
from TestReasults import result_v

data_qb = 8  # Set to ceil(log_2(image.CropSize)) hardcoded as 8 since image crop is 16x16
anc_qb = 1  # This is the auxiliary qbit.
total_qb = data_qb + anc_qb

H_SIZE = 16
V_SIZE = 16

def plot_image(img, title: str):
    plt.title(title)
    plt.imshow(img, extent=[0, img.shape[0], img.shape[1], 0], cmap='viridis')
    plt.xticks([])
    plt.yticks([])
    # A blocking request to display the figure(s) loaded. Block ends when user closes figure(s)
    # Will show glitchy overlap if mutable figures are made before show is called
    plt.show()


def plot_chunks(chunks):
    fig, axs = plt.subplots(256//H_SIZE,256//V_SIZE)
    index = 0
    for v in range(256//V_SIZE):
        for h in range(256//H_SIZE):
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

results_h = []
results_v = []
#TODO combine from here to MARK into one loop
#for each circuit we have
for i in range(243):
    counts = {f'{k:0{total_qb}b}': 0.0 for k in range(2 ** total_qb)}  #create binaries

    # Transfer all known values form experiment results to dic
    for k, v in result_h[i].items():
        counts[k] = v * 255  # And convert from probability to color int between 0-255

    results_h.append(counts)


for i in range(243):
    counts = {f'{k:0{total_qb}b}': 0.0 for k in range(2 ** total_qb)}  #create binaries

    # Transfer all known values form experiment results to dic
    for k, v in result_v[i].items():
        counts[k] = v * 255  # And convert from probability to color int between 0-255

    results_v.append(counts)


edge_scan_h = []
edge_scan_v = []
# Extract odd numbered states for each chunk. (maybe do it in the mapping above to save time?)
#do for each h circuit
for rh in range(len(results_h)):
    edge_scan_h.append(np.array([results_h[rh][f'{2 * i + 1:0{total_qb}b}'] for i in range(2 ** data_qb)]).reshape(H_SIZE, V_SIZE))

#do for each v circuit
for rv in range(len(results_v)):
    edge_scan_v.append(np.array([results_v[rv][f'{2 * i + 1:0{total_qb}b}'] for i in range(2 ** data_qb)]).reshape(H_SIZE, V_SIZE).T)
#MARK
edge_detected_image = []
# Add together the H and V of each chunk.
for i in range(len(edge_scan_h)):
    edge_detected_image.append(edge_scan_h[i] + edge_scan_v[i])


# Stitch the chunks back into one image.
# first make empty image
ed_image = Image.new('L', (256, 256))  # L mode is int between 0-255
ed_image_chunks = []
res = 0 # index of curr result
#loop for upper left box for each chunk
for i in range(len(is_empty)):
    # if there was data that was processed
    if not is_empty[i]:
        # paste it in to the image
        ULBox = ((i*16)%256, i//16*16)  # find upper left cords of chunk based off of chunk index
        ed_image.paste(Image.fromarray(edge_detected_image[res], mode='L'), box=ULBox)  # paste 16x16 chunk
        ed_image_chunks.append(edge_detected_image[res])
        res += 1 # move res to next result
    # If not then leave as default black
    else:
        ed_image_chunks.append(np.zeros((16,16)))

plot_chunks(ed_image_chunks)

# Plot edge detected image.
plot_image(np.array(ed_image), 'Full Edge Detected Image')