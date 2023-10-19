
# p1 = (x1, y1)
# p2 = (x2, y2)
def get_cropped_dataset(p1, p2):
    # make a tmp folder with a name that has the cropping in the title
    # make sub-folders for the labels
    # for each photo in Post-Proccessed_Images
        # load it
        # crop it
        # save it to the tmp directory under its proper label
    # return the name of the tmp directory
    pass

def main():
    # TODO for the croppings only loop over the bottom half of the image, so we don't waste time
    # for each cropping
        # check that it can be cleanly chopped up into 16x16's
        # crop imgs and get the tmp folder where they are stored
        # run trianing on tmp folder
        # if worse than the best, then delete the tmp folder
    pass
