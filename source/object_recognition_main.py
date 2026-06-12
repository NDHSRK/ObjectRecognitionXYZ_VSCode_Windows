from SampleRecognition import SampleRecognition
from ImageUtils import ImageUtils
import argparse
import cv2
import os
from datetime import datetime

def main():
    # Construct the argument parser and parse the arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str)  # e.g. \files\images\blue_sample.png
    ap.add_argument("--alliance", type=str)
    args = vars(ap.parse_args())

    # load the image
    IMAGE_SUBDIRECTORY = "\\files\\images\\" # const
    image_directory = os.getcwd() + IMAGE_SUBDIRECTORY
    image_filename = args["image"]
    image_full_path = image_directory + image_filename
    src = cv2.imread(image_full_path)
    if src is None:
        print('File not found')
        return

    ##TODO Currently this project uses a full image.
    # To support an ROI you need command line switches
    # for the ROI values x origin y origin, width, height.
    # Then you need to call ImageUtils.pre_process_image().

    # For writing output image files with a timestamp in
    # the filename.
    # Format: YYYY-MM-DD HH:MM:SS
    now = datetime.now()
    formatted_timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    output_file_preamble = ImageUtils.create_output_file_preamble(image_directory, image_filename, formatted_timestamp)

    # This project has a generic name, ObjectRecognitionXYZ,
    # and is intended to be merged into the project
    # CalculateXYZ as a replacement for the object recognition
    # that is part of Paco Garcia's open source project. But
    # the current project makes use of the work done on the
    # recognition of red and blue "samples" from the 2025 FTC
    # IntoTheDeep game.

    alliance = args["alliance"]
    alliance_instance = SampleRecognition.Alliance[alliance]

    try:
        recognition = SampleRecognition(alliance_instance.value, output_file_preamble)
        ret_val = recognition.perform_recognition(src)
        print(ret_val.status)

        if ret_val.status != SampleRecognition.SampleRecognitionReturn.RecognitionStatus.FAILURE:
            for one_object in ret_val.recognized_objects:
                # Print out the coordinates of each object's centerpoint
                # and its FTC angle.
                print("Recognized object with center x " + str(one_object.center_x) +
                ", center y " + str(one_object.center_y) +
                ", ftc angle " + str(one_object.ftc_angle))
    except Exception as e:
        # For debugging print out information from the exception.
        print(repr(e))

if __name__ == "__main__":
    main()