import math
from math import atan2, degrees

import cv2
import numpy as np


def gradient(pt1, pt2):
    return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])


def __get_distance_between_two_points(point_1, point_2):
    distance = math.sqrt(((point_1[0] - point_2[0]) ** 2) + ((point_1[1] - point_2[1]) ** 2))
    return distance


def draw_line_between_two_points(image, point1, point_2, line_color=(0, 0, 255), thickness=2):
    cv2.line(image, point1, point_2, line_color, thickness)


def __get_angle_between_three_points_0_360(point_1, point_2, point_3):
    """

    1-#
      #
      #
    2-# #  # # #  # -3

    :param point_1: 3
    :param point_2: 2
    :param point_3: 1
    :return:
    """
    x1, y1 = point_1
    x2, y2 = point_2
    x3, y3 = point_3
    deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
    return int(deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2))


def __get_angle_between_three_points_0_180(point_1, point_2, point_3):
    """
    1-#
      #
      #
    2-# #  # # #  # -3

    :param point_1: 2
    :param point_2: 1
    :param point_3: 3
    :return:
    """
    m1 = gradient(point_1, point_2)
    m2 = gradient(point_1, point_3)
    angR = math.atan((m2 - m1) / (1 + (m2 * m1)))
    angD = round(math.degrees(angR))

    return int(angD)


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged
