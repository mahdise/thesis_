import math

import cv2

from Development.DetectArrow.arrow_detection_pipeline_final import get_arrow_on_top


# from Development.DetectArrow.privious_test_for_arrow_detction.main_controller_detect_arrow.detect_arrow_front import get_arrow_front


def gradient(pt1, pt2):
    return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])


def get_orinatation_0_180_final_version(c, img, c_front, num_boxs=None, x0=None, y0=None):
    # need to see
    # https://stackoverflow.com/questions/64860785/opencv-using-canny-and-shi-tomasi-to-detect-round-corners-of-a-playing-card

    # Get only rectangles given exceeding area
    approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
    tmp_img = img.copy()

    # cv2.drawContours(tmp_img, [c], 0, (0, 255, 255), 6)
    # cv2.imshow('Contour Borders', tmp_img)

    # cv2.drawContours(tmp_img, [c], 0, (255, 0, 255), -1)
    # cv2.imshow('Contour Filled', tmp_img)

    # # Make a hull arround the contour and draw it on the original image
    # mask = np.zeros((img.shape[:2]), np.uint8)
    # hull = cv2.convexHull(c)
    # cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)
    # cv2.imshow('Convex Hull Mask', mask)

    # Draw minimum area rectangle
    # rect = cv2.minAreaRect(c)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # cv2.drawContours(tmp_img, [box], 0, (0, 0, 255), 2)
    # cv2.imshow('Minimum Area Rectangle', tmp_img)

    # Draw bounding rectangle
    tmp_img = img.copy()
    x, y, w, h = cv2.boundingRect(c)
    # crop image for detect arrow on top
    detect_center_x = x
    detect_center_y = y
    detect_center_w = w
    detect_center_h = h
    arrow_detect_image = tmp_img[y:(y + h), x:(x + w)]
    # cv2.rectangle(tmp_img, (x, y), (x + w, y + h), (80, 127, 257), 2)
    # cv2.imshow('Bounding Rectangle', tmp_img)

    # draw bounding box for front

    # tmp_img = img.copy()
    # x, y, w, h = cv2.boundingRect(c_front)
    # crop image for detect arrow on top
    # arrow_detect_image_front = tmp_img[y:(y+h), x:(x+w)]

    # cv2.rectangle(tmp_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow('Bounding Rectangle front', tmp_img)

    # Bounding Rectangle and Minimum Area Rectangle
    # tmp_img = img.copy()
    # rect = cv2.minAreaRect(c)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # cv2.drawContours(tmp_img, [box], 0, (0, 0, 255), 2)
    # x, y, w, h = cv2.boundingRect(c)
    # cv2.rectangle(tmp_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow('Bounding Rectangle and minimum area', tmp_img)

    # determine the most extreme points along the contour
    # https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
    tmp_img = img.copy()
    extLeft = list(c[c[:, :, 0].argmin()][0])
    extRight = list(c[c[:, :, 0].argmax()][0])
    extTop = list(c[c[:, :, 1].argmin()][0])
    extBot = list(c[c[:, :, 1].argmax()][0])

    # red
    cv2.circle(tmp_img, extLeft, 8, (0, 0, 170), -1)
    # # green
    cv2.circle(tmp_img, extRight, 8, (0, 170, 0), -1)
    # # blue
    cv2.circle(tmp_img, extTop, 8, (170, 0, 0), -1)
    # # grey
    cv2.circle(tmp_img, extBot, 8, (85, 85, 85), -1)

    # print("Corner Points: ", extLeft, extRight, extTop, extBot)

    ###### Start Calculation ########################

    # step 01 : Check arrow on top or not
    result_for_top_arrow = __get_arrow_on_top(arrow_detect_image, c, tmp_img, num_boxs)
    # result_for_top_arrow = __get_arrow_on_front(arrow_detect_image_front, c)

    # calculate center and draw
    center_x = int((detect_center_x + (detect_center_x + detect_center_w)) / 2)
    center_y = int((detect_center_y + (detect_center_y + detect_center_h)) / 2)
    cv2.circle(tmp_img, [center_x, center_y], 10, (255, 255, 0), cv2.FILLED)

    # calculate thrid point
    pt3 = [(x + w), center_y]
    # cv2.circle(tmp_img, pt3, 10, (153, 51, 255), cv2.FILLED)

    # all point for calculate angle
    pt1 = [center_x, center_y]
    pt2 = extTop

    # try:
    #     # draw line to visulize
    #     cv2.line(tmp_img, extLeft, extBot, (0, 0, 128),2)
    #     cv2.line(tmp_img, extLeft, extTop, (0, 0,128),2)
    #     cv2.line(tmp_img, extRight, extBot, (0, 0, 128),2)
    #     cv2.line(tmp_img, extRight, extTop, (0, 0, 128),2)
    #     cv2.line(tmp_img, pt1, pt3, (0, 0, 255), 2)
    #     cv2.line(tmp_img, pt1, pt2, (0, 0, 255), 2)
    #
    #
    # except:
    #     pass

    # calculate angle
    m1 = gradient(pt1, pt2)
    m2 = gradient(pt1, pt3)
    angR = math.atan((m2 - m1) / (1 + (m2 * m1)))

    # sometimes here come float none
    try:
        angD = round(math.degrees(angR))
    except:
        angD = 0

    if angD < 0:
        angD = 180 - abs(angD)

    angle = angD
    if 100 > angle > 85:
        print("Group :", "Right")
        angle_type = "Right"
    elif 50 < angle < 85:
        print("Group :", "Acute")
        angle_type = "Acute"

    elif 40 < angle < 50 or 130 < angle < 150:
        print("Group :", "Straight")
        angle_type = "Straight"


    elif 130 > angle > 100:
        print("Group :", "Obtuse")
        angle_type = "Obtuse"



    else:
        print("Group : ", None)
        angle_type = None

    extLeft = tuple([extLeft[0] + x0, extLeft[1] + y0])
    extRight = tuple([extRight[0] + x0, extRight[1] + y0])
    extTop = tuple([extTop[0] + x0, extTop[1] + y0])
    extBot = tuple([extBot[0] + x0, extBot[1] + y0])

    pt1_adjust = [center_x + x0, center_y + y0]
    pt2_adjust = [(x + w) + x0, center_y + y0]
    pt3_adjust = extTop
    final_result = {
        "angle": angle,
        "angle_type": angle_type,
        "center": [center_x + x0, center_y + y0],
        "result_for_top_arrow": result_for_top_arrow,
        "line_1": [extLeft, extBot],
        "line_2": [extLeft, extTop],
        "line_3": [extRight, extBot],
        "line_4": [extRight, extTop],
        "line_5": [tuple(pt1_adjust), tuple(pt3_adjust)],
        "line_6": [tuple(pt1_adjust), tuple(pt2_adjust)],

    }

    return final_result


def __get_arrow_on_top(image, c, tmp_img, num_box):
    scaleX = 0.6
    scaleY = 0.6
    scaleUp = cv2.resize(image, None, fx=scaleX * 6, fy=scaleY * 6, interpolation=cv2.INTER_CUBIC)
    result_arrow_on_top = get_arrow_on_top(scaleUp, tmp_img, num_box)

    return result_arrow_on_top


def __get_arrow_on_front(image, c):
    result_arrow_on_top = None
    scaleX = 0.6
    scaleY = 0.6
    scaleUp = cv2.resize(image, None, fx=scaleX * 5, fy=scaleY * 5, interpolation=cv2.INTER_CUBIC)
    # result_arrow_on_top = get_arrow_front(scaleUp)

    return {"result_top": result_arrow_on_top}


def __get_angle(image, x1y1=None, x2y2=None, num_boxes=None, x0=None, y0=None):
    """
    lower_hsv = [0, 70, 56]
    upper_hsv = [22, 251, 152]
    :param image:
    :param x1y1:
    :param x2y2:
    :return:
    """

    result_data = dict()

    try:
        # crop = image[x1y1[1]:x2y2[1], x1y1[0]:x2y2[0]]
        crop = image[x1y1[1] - 15:x2y2[1] - 15, x1y1[0] - 15:x2y2[0] + 15]

        x0 = x0 - 15
        y0 = y0 - 15

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(hsv, (0, 70, 56), (22, 251, 152))
        mask = cv2.inRange(hsv, (0, 37, 143), (23, 113, 255))
        # mask = cv2.inRange(hsv, (0, 37, 143), (23, 144, 255))
        mask_front = cv2.inRange(hsv, (0, 17, 70), (23, 139, 160))

        # output = cv2.bitwise_and(crop, crop, mask=mask)
        ret, thresh = cv2.threshold(mask, 40, 255, 0)
        ret, thresh_front = cv2.threshold(mask_front, 40, 255, 0)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_front, hierarchy = cv2.findContours(thresh_front, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) != 0:
            # find the biggest countour (c) by the area
            c = max(contours, key=cv2.contourArea)
            c_front = max(contours_front, key=cv2.contourArea)
            # cv2.drawContours(output, c, -1, 255, 3)
            # cv2.drawContours(image, [c], -1, (0, 255, 0), 3)
            # cv2.drawContours(image, [c_front], -1, (183, 33, 133), 3)

            result_data = get_orinatation_0_180_final_version(c, crop, c_front, num_boxs=num_boxes, x0=x0, y0=y0)

        # cv2.imshow("Result", np.hstack([crop, output]))
        # cv2.imshow("Result", image)
    except:
        print("Wait for calculate angle")

        try:

            crop = image[x1y1[1]:x2y2[1], x1y1[0]:x2y2[0]]

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            # mask = cv2.inRange(hsv, (0, 70, 56), (22, 251, 152))
            mask = cv2.inRange(hsv, (0, 37, 143), (23, 113, 255))
            mask_front = cv2.inRange(hsv, (0, 17, 70), (23, 139, 160))

            # output = cv2.bitwise_and(crop, crop, mask=mask)
            ret, thresh = cv2.threshold(mask, 40, 255, 0)
            ret, thresh_front = cv2.threshold(mask_front, 40, 255, 0)

            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours_front, hierarchy = cv2.findContours(thresh_front, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0:
                # find the biggest countour (c) by the area
                c = max(contours, key=cv2.contourArea)
                c_front = max(contours_front, key=cv2.contourArea)
                # cv2.drawContours(output, c, -1, 255, 3)
                # cv2.drawContours(image, [c], -1, (0, 255, 0), 3)
                # cv2.drawContours(image, [c_front], -1, (183, 33, 133), 3)

                result_data = get_orinatation_0_180_final_version(c, crop, c_front,
                                                                  num_boxs=num_boxes, x0=x0, y0=y0)
            # cv2.imshow("Result", image)

        except:
            crop = image

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            # mask = cv2.inRange(hsv, (0, 70, 56), (22, 251, 152))
            mask = cv2.inRange(hsv, (0, 37, 143), (23, 113, 255))
            mask_front = cv2.inRange(hsv, (0, 17, 70), (23, 139, 160))

            # output = cv2.bitwise_and(crop, crop, mask=mask)
            ret, thresh = cv2.threshold(mask, 40, 255, 0)
            ret, thresh_front = cv2.threshold(mask_front, 40, 255, 0)

            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours_front, hierarchy = cv2.findContours(thresh_front, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0:
                # find the biggest countour (c) by the area
                c = max(contours, key=cv2.contourArea)
                c_front = max(contours_front, key=cv2.contourArea)
                # cv2.drawContours(output, c, -1, 255, 3)
                # cv2.drawContours(image, [c], -1, (0, 255, 0), 3)
                # cv2.drawContours(image, [c_front], -1, (183, 33, 133), 3)

                result_data = get_orinatation_0_180_final_version(c, crop, c_front,
                                                                  num_boxs=num_boxes, x0=0, y0=0)

    return result_data
