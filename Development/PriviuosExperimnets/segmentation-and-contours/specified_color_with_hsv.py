# Ref: https://stackoverflow.com/questions/44588279/find-and-draw-the-largest-contour-in-opencv-on-a-specific-color-python

import numpy as np
import cv2
from math import atan2, cos, sin, sqrt, pi

def drawAxis(img, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)

    ## [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
    ## [visualization1]

def getOrientation(pts, img):
    print("dddddddddddddddddddddd")
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))

    ## [pca]

    ## [visualization]
    # Draw the principal components
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (
        cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
        cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (
        cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
        cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])

    drawAxis(img, cntr, p1, (0, 153, 0), 1)
    drawAxis(img, cntr, p2, (0, 153, 0), 5)

    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    print("angle::::", angle)

    ## [visualization]p
    print("degree : ", np.rad2deg(angle))

    # Label with the rotation angle
    label = "  Ro " + str(-int(np.rad2deg(angle)) - 90) + " de"
    print("After calculate angle",str(-int(np.rad2deg(angle)) - 90))
    # textbox = cv.rectangle(img, (cntr[0], cntr[1] - 25), (cntr[0] + 250, cntr[1] + 10), (255, 255, 255), -1)
    # cv.putText(img, label, (cntr[0], cntr[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

    return angle


# load the image
image = cv2.imread("/home/mahdiislam/Mahdi/Biba/others_repo/Object-Detection-API/real_time.jpg", 1)
# (hMin = 0 , sMin = 70, vMin = 69- 56), (hMax = 22 , sMax = 251, vMax = 152)
#
scale_percent = 60 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
image = resized
lower_hsv = [0,70,56]
upper_hsv = [22,251,152]
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv,(0, 70, 56), (22, 251, 152) )



# red color boundaries [B, G, R]
# lower = [1, 0, 20]
# upper = [60, 40, 220]
#
# # create NumPy arrays from the boundaries
# lower = np.array(lower, dtype="uint8")
# upper = np.array(upper, dtype="uint8")
#
# # find the colors within the specified boundaries and apply
# # the mask
# mask = cv2.inRange(image, lower, upper)
output = cv2.bitwise_and(image, image, mask=mask)

ret,thresh = cv2.threshold(mask, 40, 255, 0)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)



if len(contours) != 0:
    # draw in blue the contours that were founded

    # find the biggest countour (c) by the area
    c = max(contours, key = cv2.contourArea)
    print(c)
    cv2.drawContours(output, c, -1, 255, 3)

    getOrientation(c, image)

    x,y,w,h = cv2.boundingRect(c)

    # draw the biggest contour (c) in green
    # cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)

# show the images
cv2.imshow("Result", np.hstack([image, output]))

cv2.waitKey(0)


