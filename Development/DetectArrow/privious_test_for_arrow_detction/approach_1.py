import cv2
import numpy as np


class DetectArrowWithCanny:
    def __init__(self, image):
        # print(len(image))

        result = ".............Not yet........."
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 20)
        if lines is not None:
            print("len of lines")
            print(len(lines))


        left = [0, 0]
        right = [0, 0]
        up = [0, 0]
        down = [0, 0]

        if lines is not None:

            for object in lines:
                theta = object[0][1]
                rho = object[0][0]

                # cases for right/left arrows
                if ((np.round(theta, 2)) >= 1.0 and (np.round(theta, 2)) <= 1.1) or (
                        (np.round(theta, 2)) >= 2.0 and (np.round(theta, 2)) <= 2.1):
                    if (rho >= 20 and rho <= 30):
                        left[0] += 1
                    elif (rho >= 60 and rho <= 65):
                        left[1] += 1
                    elif (rho >= -73 and rho <= -57):
                        right[0] += 1
                    elif (rho >= 148 and rho <= 176):
                        right[1] += 1

                # cases for up/down arrows
                elif ((np.round(theta, 2)) >= 0.4 and (np.round(theta, 2)) <= 0.6) or (
                        (np.round(theta, 2)) >= 2.6 and (np.round(theta, 2)) <= 2.7):

                    print("...rho,,,,",rho)
                    if (rho >= -63 and rho <= -15):
                        up[0] += 1
                    elif (rho >= 67 and rho <= 74):
                        down[1] += 1
                        up[1] += 1
                    elif (rho >= 160 and rho <= 171):
                        down[0] += 1

        if left[0] >= 1 and left[1] >= 1:
            result = "left"
            print("left")
        elif right[0] >= 1 and right[1] >= 1:
            # print("right")
            result = "right"

        elif up[0] >= 1 and up[1] >= 1:
            # print("up")
            result = "up"

        elif down[0] >= 1 and down[1] >= 1:
            # print("down")
            result = "down"

        if result in ["left", "right", "up", "down"]:
            print(left, right, up, down)
        #     cv2.imshow("Arrow", edges)

        print("final result : ", result)
        cv2.imshow("canny'd image", edges)

