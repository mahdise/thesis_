import cv2
import numpy as np
import math

class DetectOrientation:
    def __init__(self, image):
        # bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # create sliders for variables
        l_v = cv2.getTrackbarPos("threshold", "Trackbars")
        u_v = cv2.getTrackbarPos("Houghlines", "Trackbars")

        # convert frame to Black and White
        bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(bw, 100, 200)


        cv2.imshow('image_orientaion', edges)

        # convert Black and White to binary image
        ret, thresh4 = cv2.threshold(bw, l_v, 255, cv2.THRESH_BINARY)
        thresh = cv2.threshold(bw, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # find the contours in thresh4
        contours, hierarchy = cv2.findContours(thresh4, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            # Obtain bounding box coordinates and draw rectangle
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)

            # Find center coordinate and draw center point
            M = cv2.moments(c)
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(image, (cx, cy), 2, (36, 255, 12), -1)
            except:
                pass

        cv2.imshow('image', image)