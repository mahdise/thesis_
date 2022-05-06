# Ref: https://stackoverflow.com/questions/19222343/filling-contours-with-opencv-python
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Lets first create a contour to use in example
image = cv2.imread('/home/mahdiislam/Mahdi/Biba/others_repo/Object-Detection-API/savedImage.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (3, 3), 0)
im_bw = cv2.Canny(gray, 225, 250)

cir = np.zeros((255,255))

res = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = res[-2] # for cv2 v3 and v4+ compatibility
# An open circle; the points are in contours[0]
plt.figure()
plt.imshow(im_bw)
#
# # Option 1: Using fillPoly
img_pl = np.zeros((255,255))
cv2.fillPoly(img_pl,pts=contours,color=(255,255,255))
plt.figure()
plt.imshow(img_pl)

# Option 2: Using drawContours
img_c = np.zeros((255,255))
cv2.drawContours(img_c, contours, contourIdx=-1, color=(255,255,255),thickness=-1)
plt.figure()
plt.imshow(img_c)
#
plt.show()