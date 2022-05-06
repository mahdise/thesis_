import cv2
import math
from math import atan2, degrees

path = '/DataTestReults/FieldTest/RawImage/0033_top_cam.png'
img = cv2.imread(path)
pointsList = []
def angle_between(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
    return deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)



def mousePoints(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        size = len(pointsList)
        if size != 0 and size % 3 != 0:
            cv2.line(img, tuple(pointsList[round((size-1) / 3) * 3]), (x, y), (0, 0, 255), 2)
        cv2.circle(img, (x, y), 5, (0, 0, 255), cv2.FILLED)
        pointsList.append([x, y])


def gradient(pt1, pt2):
    return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])


def getAngle(pointsList):
    pt1, pt2, pt3 = pointsList[-3:]

    res = angle_between( pt1, pt2, pt3)
    m1 = gradient(pt1, pt2)
    m2 = gradient(pt1, pt3)
    angR = math.atan((m2 - m1) / (1 + (m2 * m1)))
    angD = round(math.degrees(angR))

    print("res...........",res)

    if angD<0:
        angD = 180-abs(angD)

    print("angD", angD)

    cv2.putText(img, str(angD)+ "/"+str(res), (pt1[0] - 40, pt1[1] - 20), cv2.FONT_HERSHEY_COMPLEX,
                1.5, (0, 0, 255), 2)


while True:

    if len(pointsList) % 3 == 0 and len(pointsList) != 0:
        getAngle(pointsList)

    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', mousePoints)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pointsList = []
        img = cv2.imread(path)
