# ref:

import cv2
import numpy as np

def __get_angle_2(image, x1y1=None, x2y2=None):
    crop = image
    # y1-y2, x1-x2
    # crop = image[x1y1[1] - 10:x2y2[1] + 10, x1y1[0] - 30:x2y2[0] + 30]

    (h, w) = crop.shape[:2]
    cv2.circle(crop, (w // 2, h // 2), 7, (255, 0, 0), -1)
    # verticale line
    # cv2.line(crop, (w // 2, 0), (w // 2, h), (0, 255, 0), 4)
    # horizental line
    # cv2.line(crop, (0, h // 2), (w, h // 2), (0, 255, 0), 4)
    ###########
    # gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # edged = cv2.Canny(gray, 50, 100)
    # edged = cv2.dilate(edged, None, iterations=1)
    # edged = cv2.erode(edged, None, iterations=1)

    #######

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    im_bw = cv2.Canny(gray, 225, 250)

    # im_bw = auto_canny(gray)

    cv2.imwrite(
        "/home/mahdiislam/Mahdi/Biba/others_repo/Object-Detection-API/image_output_file/am-bounding-box-canny.jpg",
        im_bw)
    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow('canny edges after contouring', im_bw)
    # cv2.waitKey(0)
    print(contours)
    print('Numbers of contours found=' + str(len(contours)))
    cv2.drawContours(image, contours, -1, (0, 255, 0),thickness=-1)
    cv2.imshow('contours', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #
    # print("############################################################")
    #
    # for c in contours:
    #     # k = cv2.isContourConvex(c)
    #     # print(k)
    #     cv2.drawContours(crop, c, 0, (0, 255, 0), 3)
    #     rect = cv2.minAreaRect(c)
    #     box = cv2.boxPoints(rect)
    #     box = np.int0(box)
    #     cv2.drawContours(crop, [box], 0, (0, 0, 255), 2)
    #
    # cv2.imwrite(
    #     "/home/mahdiislam/Mahdi/Biba/others_repo/Object-Detection-API/image_output_file/am-bounding-box.jpg",
    #     crop)
    #
    #
    # print("############################################################")
    #
    # cv2.imshow("bounding-box", crop)
    # cv2.imshow("bounding-box-gray", im_bw)

    # cv2.imshow("bounding-box-edge", edged)



image = cv2.imread('/home/mahdiislam/Mahdi/Biba/others_repo/Object-Detection-API/savedImage.jpg')

__get_angle_2(image)