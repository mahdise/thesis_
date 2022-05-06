import cv2 as cv
from math import atan2, cos, sin, sqrt, pi
import numpy as np


def drawAxis(img, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)

    ## [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
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
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)

    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))

    ## [pca]

    ## [visualization]
    # Draw the principal components
    cv.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (
        cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
        cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (
        cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
        cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])

    # drawAxis(img, cntr, p1, (10, 0, 0), 1)
    # drawAxis(img, cntr, p2, (0, 0, 10), 5)

    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    print("angle::::", angle)

    ## [visualization]p
    print("degree : ", np.rad2deg(angle))

    # Label with the rotation angle
    label = "  Ro " + str(-int(np.rad2deg(angle)) - 90) + " de"
    print(str(-int(np.rad2deg(angle)) - 90))
    # textbox = cv.rectangle(img, (cntr[0], cntr[1] - 25), (cntr[0] + 250, cntr[1] + 10), (255, 255, 255), -1)
    # cv.putText(img, label, (cntr[0], cntr[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

    return angle


# Load the image
img = cv.imread("savedImage.jpg")

# Was the image there?
if img is None:
    print("Error: File not found")
    exit(0)

cv.imshow('Input Image', img)

# Convert image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

black_white = cv.subtract(255, img)

im_bw = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
blur = cv.GaussianBlur(im_bw, (5, 5), 0)
im_bw = cv.Canny(blur, 10, 90)

cv.imshow('Input Image gray', im_bw)


im_bw = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
blur = cv.GaussianBlur(im_bw, (5, 5), 0)
im_bw = cv.Canny(blur, 10, 90)

# Convert image to binary
_, bw = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

# Find all the contours in the thresholded image
contours, _ = cv.findContours(im_bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

x = 1

area_lisrt= list()
for i, c in enumerate(contours):

    # Calculate the area of each contour
    area = cv.contourArea(c)
    print(area)
    area_lisrt.append(area)



    # Ignore contours that are too small or too large
    # if area < 3700 or 100000 < area:
    #     continue
    if x == 1:
        # pass
        # Draw each contour only for visualisation purposes
        cv.drawContours(img, contours, i, (0, 0, 255), 2)

        # Find the orientation of each shape

        # getOrientation(c, img)

    # x += 1

max_area =max(area_lisrt)

for c in contours:
    area = cv.contourArea(c)

    # if area == max_area:
    cv.drawContours(img, c, -1, (0, 255, 255), 3)


print("max_area", max_area)
cv.imshow('Output Image', img)
cv.waitKey(0)
cv.destroyAllWindows()

# Save the output image to the current directory
cv.imwrite("/home/mahdiislam/Mahdi/Biba/iris-parcel-orientation-detection/Development/DetectOrientation/output_img_te.jpg", img)
