# import the necessary packages
import numpy as np
import cv2
from enum import Enum
from ImageUtils import ImageUtils


#################################################################
# SampleParameters
#################################################################

# Python grayscale threshold/HSV inRange values derived from
# IJThresholdTester or Gimp.
class SampleParameters:
    ##!! As a demonstration, for the yellow sample use the
    # green rgb channel and grayscale threshold.
    GREEN_CHANNEL_THRESHOLD_LOW = 160  # LRS03081735... Regional

    # hsv for blue
    BLUE_HSV_HUE_LOW = 110
    BLUE_HSV_HUE_HIGH = 125
    BLUE_HSV_SAT_THRESHOLD_LOW = 50
    BLUE_HSV_VAL_THRESHOLD_LOW = 100

    # hsv for red
    RED_HSV_HUE_LOW = 170
    RED_HSV_HUE_HIGH = 5
    RED_HSV_SAT_THRESHOLD_LOW = 125
    RED_HSV_VAL_THRESHOLD_LOW = 50

    MIN_SAMPLE_AREA = 9000.0
    MAX_SAMPLE_AREA = 21000.0

    MIN_SAMPLE_ASPECT_RATIO = 1.6
    MAX_SAMPLE_ASPECT_RATIO = 3.0


#################################################################
# OpenCVRotatedRect
#################################################################
# Class for the public attributes in an OpenCV RotatedRect,
# created so that we can reference them by name. The attributes
# are defined in the c++ documentation as:

# float 	angle
# 	returns the rotation angle. When the angle is 0, 90, 180, 270 etc., the rectangle becomes an up-right rectangle.

# Point2f 	center
# 	returns the rectangle mass center

# Size2f 	size
# 	returns width and height of the rectangle

# In Python these attributes are represented as:
# (center(x, y), (width, height), angle of rotation)

class OpenCVRotatedRect:
    def __init__(self, opencv_rotated_rect):
        self.opencv_rotated_rect = opencv_rotated_rect
        self.center_x = opencv_rotated_rect[0][0]
        self.center_y = opencv_rotated_rect[0][1]
        self.width = opencv_rotated_rect[1][0]
        self.height = opencv_rotated_rect[1][1]
        self.angle = opencv_rotated_rect[2]


#################################################################
# SampleRecognition
#################################################################

class SampleRecognition:

    class Alliance(Enum):
        BLUE = 1
        RED = 2

    class SampleColor(Enum):
        BLUE = 0
        RED = 1
        YELLOW = 2
        NONE = 3

    class SampleOrientation(Enum):
        VERTICAL = 1
        HORIZONTAL = 2
        COUNTER_CLOCKWISE = 3
        CLOCKWISE = 4

    def __init__(self, p_alliance, p_output_filename_preamble):
        self.alliance = p_alliance
        self.output_filename_preamble = p_output_filename_preamble
        self.image_roi_height = 0.0
        self.image_roi_width = 0.0
        self.image_roi_center = (0.0, 0.0)

        self.TAG = self.__class__.__name__ # for logging

    # The final return value from perform_recognition should
    # be a class object (SampleRecognitionReturn) that
    # contains a status and a list of class objects, each of
    # which contains the x and y coordinates and the FTC angle
    # of a recognized object.
    class SampleRecognitionReturn:
        class RecognitionStatus(Enum):
            SUCCESS = 200
            PYTHON_APP_CRASH = 300
            IMAGE_NOT_AVAILABLE = 450
            FAILURE = 500

        class ObjectCoordinatesAndAngle:
            def __init__(self, center_x, center_y, ftc_angle):
                self.center_x = center_x
                self.center_y = center_y
                self.ftc_angle = ftc_angle

        def __init__(self, status, recognized_objects):
            self.status = status
            self.recognized_objects = recognized_objects

    def perform_recognition(self, image):
        self.image_roi_height, self.image_roi_width = image.shape[:2]
        self.image_roi_center = (self.image_roi_width / 2.0, self.image_roi_height / 2.0)

        # For the ObjectRecognitionXYZ prototype recognize
        # samples with the same color as the alliance.
        hsv_roi = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if self.alliance == SampleRecognition.Alliance.BLUE:
            thresholded = ImageUtils.apply_inRange(hsv_roi,
                                                   SampleParameters.BLUE_HSV_HUE_LOW,
                                                   SampleParameters.BLUE_HSV_HUE_HIGH,
                                                   SampleParameters.BLUE_HSV_SAT_THRESHOLD_LOW,
                                                   SampleParameters.BLUE_HSV_VAL_THRESHOLD_LOW)
        else:
            thresholded = ImageUtils.apply_inRange(hsv_roi,
                                                   SampleParameters.RED_HSV_HUE_LOW,
                                                   SampleParameters.RED_HSV_HUE_HIGH,
                                                   SampleParameters.RED_HSV_SAT_THRESHOLD_LOW,
                                                   SampleParameters.RED_HSV_VAL_THRESHOLD_LOW)

        # Filter out zero-area contours.
        filtered = ImageUtils.filter_contours(thresholded, self.image_roi_height, self.image_roi_width)
        if not filtered:
            print("No valid contours found")
            return self.SampleRecognitionReturn(self.SampleRecognitionReturn.RecognitionStatus.FAILURE, [])

        rectangles = self.enclose_and_filter_samples(filtered, self.image_roi_height, self.image_roi_width,
                                                     SampleParameters.MIN_SAMPLE_AREA / 2.0,
                                                     SampleParameters.MAX_SAMPLE_AREA)
        if not rectangles:
            print("No valid rotated rectangles found")
            return self.SampleRecognitionReturn(self.SampleRecognitionReturn.RecognitionStatus.FAILURE, [])

        # Iterate through the rectangles and reformat the data for the
        # final return value.
        recognized_objects = []
        show_rects = np.copy(image)
        for one_rect in rectangles:
            # From the OpenCV RotatedRect determine the orientation and
            # FTC angle of the sample.
            sample_orientation, ftc_angle = self.get_sample_orientation_and_ftc_angle(one_rect)
            print("Sample orientation: " + str(sample_orientation) + ", FTC angle " + str(ftc_angle))

            recognized_objects.append(
                SampleRecognition.SampleRecognitionReturn.ObjectCoordinatesAndAngle(one_rect.center_x, one_rect.center_y,
                                                                                    ftc_angle))
            # Draw the outlines of each rotated rectangle on
            # the original image.
            box = cv2.boxPoints(one_rect.opencv_rotated_rect)
            box = np.int32(box)

            # Draw the rectangle
            cv2.drawContours(show_rects, [box], 0, (0, 255, 0), 2)

        cv2.imshow("Recognized objects ", show_rects)
        cv2.waitKey(0)

        rects_filename = self.output_filename_preamble + "_RECT.png"
        cv2.imwrite(rects_filename, show_rects)
        print(self.TAG + " Writing " + rects_filename)

        return self.SampleRecognitionReturn(self.SampleRecognitionReturn.RecognitionStatus.SUCCESS,
                                            recognized_objects)

    # Assume that the parameter p_filtered_contours is a collection of OpenCV contours,
    # each with a non-zero area. Returns a collection of OpenCV rotated rectangles that
    # pass the area filters.
    ##TODO You can use height and width to check aspect ratio.
    @staticmethod
    def enclose_and_filter_samples(p_filtered_contours, image_height, image_width, min_area, max_area):
        filtered_rects = []
        for one_contour in p_filtered_contours:
            # Find the minimum area rectangle
            rect = cv2.minAreaRect(one_contour)
            center, dimensions, angle = rect
            rect_area = dimensions[0] * dimensions[1]
            print("Rotated rect area " + str(rect_area))
            if rect_area < min_area:
                print("Rotated rect below minimum area")
                continue

            if rect_area > max_area:
                print("Rotated rect above maximum area")
                continue

            filtered_rects.append(OpenCVRotatedRect(rect))

        return filtered_rects

    # The x and y coordinates of the points of an OpenCV RotatedRect
    # are relative to 0,0 at the viewer's upper left. The points
    # are numbered 0 to 3 in the clockwise direction, where point 0
    # is that with the lowest x coordinate. If two points have the
    # same x coordinate, which will happen if a rectangle is
    # perfectly horizontal or perfectly vertical, the point with the
    # lower of the two y coordinates becomes point 0.
    #
    # The angle of a RotatedRect is always measured from point 0 and
    # always lies between [0,90], exclusive of 0 but inclusive of 90.
    # The angle is measured from a line between point 0 and point 1
    # and a line through point 0 parallel to the left boundary of
    # the image. What is not obvious is that the corner of a rectangle
    # designated as point 0 can change with rotation.

    # For example, for a counter-clockwise rotation of a perfectly
    # horizontal or vertical rectangle, the angle starts at 90 and
    # decreases. Point 0 stays the same but then instead of reaching
    # 0 it changes to 90 and the previous point 1 becomes point 0.
    # For a clockwise rotation of a perfectly horizontal or vertical
    # rectangle, the angle starts at 90 but then, as soon as the
    # rotation starts, it changes to the first value greater than 0
    # and point 3 becomes point 0. The angle increases until it
    # reaches 90; point 0 remains the same.

    # To identify the orientation of a rectangle it is necessary to
    # look at the RotatedRectangle fields size.height and size.width.
    # The size.height of a RotatedRect is *always" the distance
    # between point 0 x and point 1 x; the size.width is always the
    # distance between point 1 y and point 2 y. In the case of
    # IntoTheDeep samples, our "width" is always the longer of these
    # two RotatedRect values, i.e. the 3.5" side of a sample.

    # The angle of both a perfectly vertical RotatedRectangle
    # and a perfectly horizontal RotatedRectangle is 90.0.
    def get_sample_orientation_and_ftc_angle(self, p_rotated_sample):
        # Always get the box points for debugging.
        box = cv2.boxPoints(p_rotated_sample.opencv_rotated_rect)
        rect_points = np.int32(box)  # Integer values for pixel indices

        if p_rotated_sample.angle == 90.0:
            if p_rotated_sample.height < p_rotated_sample.width:
                sample_orientation = self.SampleOrientation.VERTICAL
                ftc_angle = 0.0
            elif p_rotated_sample.height > p_rotated_sample.width:
                sample_orientation = self.SampleOrientation.HORIZONTAL
                ftc_angle = 90.0
            else:  # square
                sample_orientation = self.SampleOrientation.VERTICAL
                ftc_angle = 0.0

        # The RotatedRect must be angled counter-clockwise or clockwise
        elif p_rotated_sample.width > p_rotated_sample.height:
            sample_orientation = self.SampleOrientation.COUNTER_CLOCKWISE
            ftc_angle = 90.0 - p_rotated_sample.angle
        elif p_rotated_sample.width < p_rotated_sample.height:
            sample_orientation = self.SampleOrientation.CLOCKWISE
            ftc_angle = -1 * p_rotated_sample.angle
        else:
            # Got an angled square. This should not happen but we have to deal with it.
            # If point 1 y < point 0 y then the orientation is CCW, else CW.
            if round(rect_points[1][1]) < round(rect_points[0][1]):
                sample_orientation = self.SampleOrientation.COUNTER_CLOCKWISE
                ftc_angle = 90.0 - p_rotated_sample.angle
            else:
                sample_orientation = self.SampleOrientation.CLOCKWISE
                ftc_angle = -1 * p_rotated_sample.angle

        return sample_orientation, ftc_angle
