"""
This file will take all photos present in the folder and push them through the QED.
All original photos will be categorized into sub-folders here based off of tag.
All processed photos will go into the dataset folder under the sub-folder of it tag.

NOTE: ALL PHOTOS MUST HAVE THE TAG IN ITS NAME

The valid tag names are present on the Config.ini file
The location of the tag in the name is up to you.
"""
import configparser
import logging
from PIL import Image
import sys
import os
import os.path as fs
import ast

from Quantum.QED import QED


# get logger, use __name__ to prevent conflict
log = logging.getLogger(__name__)
logging.basicConfig(filename="../Latest_Run.log", filemode='w', encoding='utf-8', level=logging.INFO,
                    format="%(asctime)s : %(levelname)s : %(name)s : %(funcName)s : %(message)s",
                    datefmt='%m/%d %I:%M:%S %p')


# Create a configparser object
config = configparser.ConfigParser()
# Read an existing configuration file
config.read_file(open("./../Config.ini"))

tags = ast.literal_eval(config['NN']['DATA_TAGS'])
data_dir = config['NN']['TRAINING_DATA_DIR']

if not isinstance(tags, list):
    log.error("Unable to convert NN.DATA_TAGS to a list. Check config file")
    exit(-1)

MAX_PROCESS = int(sys.argv[1]) # Set a hard limit to the number of processed photos, for testing

def get_tag(name: str):
    for tag in tags:
        if name.lower().find(tag) != -1:
            return tag
    return None


def main():
    files = os.listdir(".")
    files.remove("Staging.py")  # Remove this file from the list, so it will not attempt to process it.
    dir_size = len(files)
    processed_count = 0  # count the num of files successfully processed

    for file in files:
        if MAX_PROCESS != -1 and processed_count == MAX_PROCESS:
            log.info("Max processed reached. Stopping...")
            break
        label = get_tag(file)
        # determine tag from name
        if (label is None) or fs.isdir(file):
            # if a tag can't be seen/isdir then skip.
            log.warning(f"File: {file} does not have a valid tag or is a directory. Skipping...")
            continue

        log.info(f"Processing photo {file} {processed_count+1}/{dir_size}")
        print(f"Processing photo {processed_count + 1}/{dir_size}")
        full_path = fs.abspath(file)
        processed = QED(full_path) # process photo and get back np.array of ed image
        if processed is not None:
            processed *= 1000000 # Scale up probabilities to ints
            # convert np.array to PIL.Image, JPEG can't use floating-point numbers so convert to 0-255 (greyscale)
            img = Image.fromarray(processed).convert("L")
            # save Image to Post-Processed/label folder
            save_path = f"{data_dir}/{label.title()}/{file}"

            try:
                img.save(save_path)  # if save_path is not a proper path it will throw an OSError
                # Move old photo into sub-folder
                os.rename(fs.abspath(file), f"{label.title()}/{file}")
            except OSError as e:
                log.error(f"Unable to save photo: {save_path}\nCheck file system to see if path is viable.")
                log.error(f"Full Error:\n{e}")
                exit(-2) # if an OSError happens it will most likely happen with the other files so exit

            processed_count += 1
            log.info(f"Photo {file} processed and saved in the {label.title()} folder.")
        else:
            log.warning(f"Processing error encountered, skipping {file}")
    # All files looped over close/clean up.
    log.info(f"Done, processed {processed_count} files.\n"
             f"Skipped {dir_size - processed_count} files for invalid name and/or processing error(s).")


if __name__ == "__main__":
    main()
