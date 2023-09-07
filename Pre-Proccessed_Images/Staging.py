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
import os
import os.path as fs

from Quantum.QED import QED


# get logger, use __name__ to prevent conflict
log = logging.getLogger(__name__)
log.info("setting up QED")

# Create a configparser object
config = configparser.ConfigParser()
# Read an existing configuration file
config.read_file(open("./../Config.ini"))

tags = config['NN']['DATA_TAGS']
data_dir = config['NN']['TRAINING_DATA_DIR']

def main():
    files = os.listdir(".")
    files.remove("Staging.py")  # Remove this file from the list, so it will not attempt to process it.
    dir_size = len(files)
    processed_count = 0  # count the num of files successfully processed
    for file in files:
        # determine tag from name
        if (file.lower() not in tags) or fs.isdir(file):
            # if a tag can't be seen/isdir then skip.
            log.warning(f"File: {file} does not have a valid tag or is a directory. Skipping...")
            continue
        label = None
        for tag in tags:
            if file.lower() in tag:
                label = tag
        # sanity error check
        if label is None:
            log.error(f"Tag scanning failed\nOffending File: {file}")
            exit(-1)

        log.info(f"Processing photo {file} {processed_count+1}/{dir_size}")
        full_path = fs.abspath(file)
        processed = QED(full_path) # process photo and get back np.array of ed image
        if processed is not None:
            # convert np.array to PIL.Image
            img = Image.fromarray(processed)
            # save Image to Post-Processed/label folder
            save_path = fs.join(fs.abspath(f"{data_dir}/{label}"), file)

            try:
                img.save(save_path)  # if save_path is not a proper path it will throw an OSError
            except OSError as e:
                log.error(f"Unable to save photo: {save_path}\nCheck file system to see if path is viable.")
                log.error(f"Full Error:\n{e}")
                exit(-2) # if an OSError happens it will most likely happen with the other files so exit

            processed_count += 1
            log.info(f"Photo {file} processed and saved in {label} folder.")
        else:
            log.warning(f"Processing error encountered, skipping {file}")
    # All files looped over close/clean up.
    log.info(f"Done, processed {processed_count} files.\n"
             f"Skipped {dir_size - processed_count} files for invalid name and/or processing error(s).")


if __name__ == "__main__":
    main()
