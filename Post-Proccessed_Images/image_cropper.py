from PIL import Image
import os


def crop_image(image_path, x1, y1, x2, y2):
    '''
    Current cropping:
        Define the cropping box coordinates (This will resize the image from 640 x 480 to 400 x 160)

        Height of image stays the same across all images y1 = 320 and y2 = 480 will yield the height of 160

        Width is dependent on left/right/straight.
            For left, crop more of the right side data. x1 = 80 and x2 = 480
            For right, crop more of the left side data. x1 = 160, x2 = 560
            For straight, crop an even amount from both sides. x1 = 120, x2 = 520
    '''

    # Getting an image from the path
    image = Image.open(image_path)

    # Crop the image according to the crop box passed in
    cropped_image = image.crop((x1, y1, x2, y2))

    # Returning the cropped image
    return cropped_image


def main():
    # Current working directory
    directory = os.getcwd()

    # Dataset locations
    dataset_left = directory + "\\Left\\"
    dataset_right = directory + "\\Right\\"
    dataset_straight = directory + "\\Straight\\"

    # Image counters to increment the naming of images
    left_image_counter = 0
    right_image_counter = 0
    straight_image_counter = 0

    # Starting left crop
    for filename in os.listdir(dataset_left):
        if filename.endswith(".jpg"):
            left_image_counter += 1
            cropped_image = crop_image(dataset_left + filename, x1=80, y1=320, x2=480, y2=480)
            cropped_image.save(
                directory + f"\\cropped_images\\Left\\cropped_image{left_image_counter}.jpg")

    # Starting right crop
    for filename in os.listdir(dataset_right):
        if filename.endswith(".jpg"):
            right_image_counter += 1
            cropped_image = crop_image(dataset_right + filename, x1=160, y1=320, x2=560, y2=480)
            cropped_image.save(
                directory + f"\\cropped_images\\Right\\cropped_image{right_image_counter}.jpg")

    # Starting straight crop
    for filename in os.listdir(dataset_straight):
        if filename.endswith(".jpg"):
            straight_image_counter += 1
            cropped_image = crop_image(dataset_straight + filename, x1=120, y1=320, x2=520, y2=480)
            cropped_image.save(
                directory + f"\\cropped_images\\Straight\\cropped_image{straight_image_counter}.jpg")


main()
