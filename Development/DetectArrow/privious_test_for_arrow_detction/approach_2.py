import cv2
import numpy as np
import itertools
from typing import List, Optional, Tuple

def normalize(
        vector: np.ndarray,
) -> np.ndarray:
    """
    :param vector: A vector
    :return: The normalized vector
    """
    return vector / np.linalg.norm(vector)

def angle(
        v1: np.ndarray,
        v2: np.ndarray,
        inner: bool=False,
) -> float:
    """
    :param v1: One vector
    :param v2: Another vector
    :param inner: Whether to calculate the inner angle
    :return: Angle between the two vectors
    """
    v1 = normalize(v1)
    v2 = normalize(v2)
    angle = np.arccos(np.clip(np.dot(v1, v2), a_min=-1, a_max=1))
    if inner:
        return angle
    cross = np.cross(v1, v2)
    return angle if cross <= 0 else np.pi * 2 - angle

def as_rad(
        angle_deg: float,
) -> float:
    """
    :param angle_deg: An angle in degrees
    :return: The angle in radians
    """
    return angle_deg * np.pi / 180

def angle_heuristic(
        desired_angle: float,
        v1: np.ndarray,
        v2: np.ndarray,
        min_angle: int = 10,
) -> float:
    """
    :param desired_angle: Desired angle in radians
    :param v1: First vector
    :param v2: Second vector
    :param min_angle: Minimum valid angle in degrees
    :return: Heuristic to minimize
    """
    min_angle = as_rad(min_angle)
    inner_angle = angle(v1, v2, True)
    if inner_angle < min_angle:
        return np.pi * 2
    return min(
        abs(desired_angle - inner_angle),
        abs(desired_angle - np.pi + inner_angle),
    )

def intersection(
        l1: np.ndarray,
        l2: np.ndarray):
    """
    Find the intersection of two lines if it exists. Lines are specified as
    pairs of points: [[x1, y1], [x2, y2]].
    This function was inspired by
    http://stackoverflow.com/a/7448287/4637060.
    :param l1: The first line
    :param l2: The second line
    :return: The intersection if it exists
    """
    l1 = l1.reshape((2, 2))
    l2 = l2.reshape((2, 2))

    x = l2[0] - l1[0]
    d1 = l1[1] - l1[0]
    d2 = l2[1] - l2[0]
    cross = np.cross(d1, d2)

    if abs(cross) < 1e-8:
        return

    t1 = (x[0] * d2[1] - x[1] * d2[0]) / cross
    return l1[0] + d1 * t1

class DetectArrowWithCanny:
    def __init__(self, image):
        # print(len(image))

        result = ".............Not yet........."
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        l_v = cv2.getTrackbarPos("threshold", "Trackbars")

        ret, thresh4 = cv2.threshold(gray, l_v, 255, cv2.THRESH_BINARY)
        self.contours, hierarchy = cv2.findContours(thresh4, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        best_combinations = []

        for contour in self.contours:
            contour = cv2.approxPolyDP(contour, 10, closed=True).squeeze()
            lines = []
            for p1, p2 in zip(contour, np.array([*contour[1:], contour[:1]])):
                lines.append(np.append(p1, p2))
            vectors = [
                (i, [x2 - x1, y2 - y1])
                for i, (x1, y1, x2, y2) in enumerate(lines)
            ]
            combinations = [
                (angle_heuristic(32, c[0][1], c[1][1]), c)
                for c in itertools.combinations(vectors, 2)
            ]
            best = min(combinations, key=lambda c: c[0])
            best_combinations.append(
                (best[0], [lines[best[1][0][0]], lines[best[1][1][0]]])
            )

        arr_lines =  min(best_combinations)[1]

        if len(arr_lines) > 2:
            arr_pos = cv2.minEnclosingCircle(
                np.append(arr_lines[0], arr_lines[1]).reshape((4, 2))
            )[0]

            arr_tip = intersection(*arr_lines)
            if arr_tip is not None:
                result = np.rint([arr_pos, arr_tip]).astype(int)


        print(result)


if __name__ == '__main__':
    img = cv2.imread(
        "/Development/priviuos_test/detect_arrow/1_arrow_down.png")

    detect_arrow = DetectArrowWithCanny(img)