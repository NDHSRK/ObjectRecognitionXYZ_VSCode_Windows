# import the necessary packages
from FTCRobotError import FTCRobotError
import numpy as np
import cv2

## 7/13/2025 copy/paste from Pycharm AutomaticThresholding project
# with modifications to contour filtering for the Limelight.
## 4/2026 Modify find_contours for ObjectRecognitionXYZ.
#################################################################
# ImageUtils.py
#################################################################
class ImageUtils:

    @staticmethod
    def create_output_file_preamble(p_image_directory, p_image_filename, p_file_date):
        # Sanity check on the file name extension.
        if not (p_image_filename.endswith(".png") or p_image_filename.endswith(".jpg")):
            raise FTCRobotError("Invalid image file name")

        # Strip the extension from the end of the file name and append an underscore and time stamp.
        index = p_image_filename.rindex(".")  # highest index, exception if not found
        filename_only = p_image_filename[:-4]
        return p_image_directory + filename_only + "_" + p_file_date

    '''
    In the OpenCV Python API (cv2), there is no native, standalone Rect class object.
    Instead, a "Rect" is defined implicitly as a standard Python tuple or list of four
    integers following the format (x, y, w, h).
    When an OpenCV function expects or returns a rectangle, it maps directly to the
    following structure:
    x: The x-coordinate of the top-left corner (horizontal position).
    y: The y-coordinate of the top-left corner (vertical position).
    w: The width of the rectangle extending to the right.
    h: The height of the rectangle extending downward.
    '''
    @staticmethod
    def pre_process_image(p_original_image, p_roi_x, p_roi_y, p_roi_width, p_roi_height):
        # Create an *independent* Region of Interest using .copy()
        roi = p_original_image[p_roi_y: p_roi_y + p_roi_height,
              p_roi_x: p_roi_x + p_roi_width].copy()
        return roi

    @staticmethod
    def apply_grayscale_threshold(p_grayscale_image, grayscale_threshold_low):
        thresh_binary_flag = cv2.THRESH_BINARY if grayscale_threshold_low >= 0 else cv2.THRESH_BINARY_INV
        _, thresholded = cv2.threshold(p_grayscale_image, grayscale_threshold_low, 255, thresh_binary_flag)
        return thresholded

    @staticmethod
    def apply_inRange(p_hsv_roi, hue_low, hue_high, sat_threshold_low, val_threshold_low):
        # Sanity check for hue.
        if not ((0 <= hue_low <= 180) and (0 <= hue_high <= 180)):
            raise Exception("Hue out of range")

        if hue_low < hue_high:  # Normal hue range.
            # Define lower and upper bounds in this way to avoid Python warnings.
            lower_bounds = np.array([hue_low, sat_threshold_low, val_threshold_low], dtype=np.uint8)
            upper_bounds = np.array([hue_high, 255, 255], dtype=np.uint8)
            thresholded = cv2.inRange(p_hsv_roi, lower_bounds, upper_bounds)
        else:
            # For a hue range from the XML file of low 170, high 10
            # the following yields two new ranges: 170 - 180 and 0 - 10.
            lower_bounds_1 = np.array([hue_low, sat_threshold_low, val_threshold_low])
            upper_bounds_1 = np.array([180, 255, 255])
            range1 = cv2.inRange(p_hsv_roi, lower_bounds_1, upper_bounds_1)

            lower_bounds_2 = np.array([0, sat_threshold_low, val_threshold_low])
            upper_bounds_2 = np.array([hue_high, 255, 255])
            range2 = cv2.inRange(p_hsv_roi, lower_bounds_2, upper_bounds_2)
            thresholded = cv2.bitwise_or(range1, range2)

        return thresholded

    @staticmethod
    def get_hue_range(p_hist, dominant_bin_index):
        # Log all non-zero histogram bins.
        min_pixel_count = np.min(p_hist)
        print("Minimum pixel count", min_pixel_count)
        for bin_index, count in enumerate(p_hist):
            if count[0] != min_pixel_count:
                print(f"Bin {bin_index}: {count[0]}")

        # Look at bins on each side of the dominant bin/hue
        # until you find one with the minimum pixel count,
        # typically 0. Be mindful of the wrap-around at 0/180.
        adjacent_bin = dominant_bin_index
        while True:
            adjacent_bin = adjacent_bin - 1
            if adjacent_bin == -1:
                adjacent_bin = 179  # crossed boundary at 0

            print("Bin " + str(adjacent_bin) + ", pixel count " + str(p_hist[adjacent_bin]))
            if p_hist[adjacent_bin] == min_pixel_count:
                print("Found minimum pixel count at bin " + str(adjacent_bin))
                hsv_hue_low = adjacent_bin
                break

        adjacent_bin = dominant_bin_index
        while True:
            adjacent_bin = adjacent_bin + 1
            if adjacent_bin == 180:
                adjacent_bin = 0  # crossed boundary at 179

            print("Bin " + str(adjacent_bin) + ", pixel count " + str(p_hist[adjacent_bin]))
            if p_hist[adjacent_bin] == min_pixel_count:
                print("Found minimum pixel count at bin " + str(adjacent_bin))
                hsv_hue_high = adjacent_bin
                break

        print("Hue low, high " + str(hsv_hue_low) + ", " + str(hsv_hue_high))
        return hsv_hue_low, hsv_hue_high

    @staticmethod
    def filter_contours(p_thresholded, image_height, image_width):
        contours, _ = cv2.findContours(p_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw on an all-black background; drawContours requires a BGR image.
        show_contours = np.zeros((image_height, image_width, 3), dtype=np.uint8)

        filtered_contours = []
        for i in range(len(contours)):
            contour_area = cv2.contourArea(contours[i])
            # oddly, some contours have zero area; test for closed contours
            # cv2.isContourConvex(contours[i]) missed a contour of 15686!
            if contour_area > 0.0:
                # Got a contour whose area is non-zero
                cv2.drawContours(show_contours, contours, i, (255, 255, 255), 2)
                #cv2.imshow("Found 1 contour ", show_contours)
                #cv2.waitKey(0)

                filtered_contours.append(contours[i])

        return filtered_contours
