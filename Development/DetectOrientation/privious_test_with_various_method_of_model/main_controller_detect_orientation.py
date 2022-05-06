import cv2
import math
import numpy as np
from Development.DetectArrow import get_arrow_on_top
from Development.DetectArrow import get_arrow_front
from Development.MainDetection.data_read_write import write_data_for_angle

def gradient(pt1, pt2):
    return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
def get_orinatation_0_180_2nd_version(c, img, c_front, file_path_read_data, num_boxs= None):

    path_for_data_raed_angle = file_path_read_data["angle_data_path"]

    # need to see
    # TODO : NEED TO SEE TO GET ANGLE
    #https://stackoverflow.com/questions/64860785/opencv-using-canny-and-shi-tomasi-to-detect-round-corners-of-a-playing-card
    ## Get only rectangles given exceeding area
    approx = cv2.approxPolyDP(c,0.01 * cv2.arcLength(c, True), True)
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
    arrow_detect_image = tmp_img[y:(y+h), x:(x+w)]
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
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    #red
    cv2.circle(tmp_img, extLeft, 8, (0, 0, 170), -1)
    #green
    cv2.circle(tmp_img, extRight, 8, (0, 170, 0), -1)
    #blue
    cv2.circle(tmp_img, extTop, 8, (170, 0, 0), -1)
    #grey
    cv2.circle(tmp_img, extBot, 8, (85, 85, 85), -1)

    # print("Corner Points: ", extLeft, extRight, extTop, extBot)

    ###### Start Calculation ########################


    # step 01 : Check arrow on top or not
    result_for_top_arrow = __get_arrow_on_top(arrow_detect_image, c, file_path_read_data, tmp_img)
    # result_for_top_arrow = __get_arrow_on_front(arrow_detect_image_front, c)

    # calculate center and draw
    center_x = int((detect_center_x + (detect_center_x+detect_center_w))/2)
    center_y = int((detect_center_y + (detect_center_y+detect_center_h))/2)
    cv2.circle(tmp_img, [center_x,center_y], 10, (255, 255, 0), cv2.FILLED)

    # calculate thrid point
    pt3 = [(x+w), center_y]
    cv2.circle(tmp_img, pt3, 10, (153, 51, 255), cv2.FILLED)

    # all point for calculate angle
    pt1 = [center_x, center_y]
    pt2 = extTop

    try:
        # draw line to visulize
        cv2.line(tmp_img, extLeft, extBot, (0, 0, 128),2)
        cv2.line(tmp_img, extLeft, extTop, (0, 0,128),2)
        cv2.line(tmp_img, extRight, extBot, (0, 0, 128),2)
        cv2.line(tmp_img, extRight, extTop, (0, 0, 128),2)
        cv2.line(tmp_img, pt1, pt3, (0, 0, 255), 2)
        cv2.line(tmp_img, pt1, pt2, (0, 0, 255), 2)


    except:
        pass



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

    write_data_for_angle(data=angle, file_path=path_for_data_raed_angle)

    label = "Ori : "+ str(angle)

    cv2.putText(tmp_img, label, (20, 30),
               cv2.FONT_HERSHEY_SIMPLEX,1, (24, 24, 228), 4, cv2.LINE_AA)

    name_of_window = 'Detect angel of box number '+str(num_boxs)
    cv2.imshow(name_of_window, tmp_img)



def __get_arrow_on_top(image, c, file_path_read_data, tmp_img):


    scaleX = 0.6
    scaleY = 0.6
    scaleUp = cv2.resize(image, None, fx=scaleX * 6, fy=scaleY * 6, interpolation=cv2.INTER_CUBIC)
    result_arrow_on_top = get_arrow_on_top(scaleUp, file_path_read_data, tmp_img)

    return {"result_top": result_arrow_on_top}

def __get_arrow_on_front(image, c):


    scaleX = 0.6
    scaleY = 0.6
    scaleUp = cv2.resize(image, None, fx=scaleX * 5, fy=scaleY * 5, interpolation=cv2.INTER_CUBIC)
    result_arrow_on_top = get_arrow_front(scaleUp)

    return {"result_top": result_arrow_on_top}
def get_orinatation_0_180(c, img):

    # need to see
    # TODO : NEED TO SEE TO GET ANGLE
    #https://stackoverflow.com/questions/64860785/opencv-using-canny-and-shi-tomasi-to-detect-round-corners-of-a-playing-card

    # cv.minAreaRect returns:
    # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # check rect box
    # red
    cv2.circle(img, box[0], 10, (0, 0, 170), cv2.FILLED)
    # green
    cv2.circle(img, box[1], 10, (0, 170, 0), cv2.FILLED)
    #blue
    cv2.circle(img, box[2], 10, (170, 0, 0), cv2.FILLED)
    # # grey
    cv2.circle(img, box[3], 10, (85, 85, 85), cv2.FILLED)

    # Retrieve the key parameters of the rotated bounding box
    center = (int(rect[0][0]), int(rect[0][1]))

    cv2.circle(img, center, 10, (255, 255, 0), cv2.FILLED)

    # Calculate angle
    pt1 = center
    pt2 = box[2]
    pt3 = [box[2][0], center[1]]

    cv2.circle(img, pt3, 10, (153, 51, 255), cv2.FILLED)

    m1 = gradient(pt1, pt2)
    m2 = gradient(pt1, pt3)
    angR = math.atan((m2 - m1) / (1 + (m2 * m1)))
    angD = round(math.degrees(angR))

    if angD<0:
        angD = 180-abs(angD)

    angle = angD
    # width = int(rect[1][0])
    # height = int(rect[1][1])
    # angle = int(rect[2])
    #
    # print("###########################################")
    # print("angle-original : ", angle)

    label = "Ori : "+ str(angle)
    # if width < height:
    #     angle = 90 - angle
    # else:
    #     angle = -angle
    # print("After calculation ",angle)
    #
    # print("###########################################")
    label = label+ " Afte_calculation : "+str(angle)
    # textbox = cv2.rectangle(img, (center[0] - 35, center[1] - 25),
    #                        (center[0] + 295, center[1] + 10), (255, 255, 255), -1)
    cv2.putText(img, label, (20, 30),
               cv2.FONT_HERSHEY_SIMPLEX,1, (24, 24, 228), 1, cv2.LINE_AA)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    cv2.imshow("Result_of_orientation_detect", img)

def __get_angle(image, x1y1=None, x2y2=None, file_path_read_data = None, num_boxes = None):

    """
    lower_hsv = [0, 70, 56]
    upper_hsv = [22, 251, 152]
    :param image:
    :param x1y1:
    :param x2y2:
    :return:
    """





    try:
        # crop = image[x1y1[1]:x2y2[1], x1y1[0]:x2y2[0]]
        crop = image[x1y1[1]-15:x2y2[1]-15, x1y1[0]-15:x2y2[0]+15]

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

            get_orinatation_0_180_2nd_version(c, crop, file_path_read_data, num_boxs=num_boxes)

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

                get_orinatation_0_180_2nd_version(c, crop, c_front, file_path_read_data, num_boxs=num_boxes)
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

                get_orinatation_0_180_2nd_version(c, crop, c_front, file_path_read_data, num_boxs=num_boxes)
            # cv2.imshow("Result", image)
