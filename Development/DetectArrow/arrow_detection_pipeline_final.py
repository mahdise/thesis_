from math import atan2, degrees

import cv2
import numpy as np

from Development.MainDetection.utils import __get_distance_between_two_points

font = cv2.FONT_HERSHEY_SIMPLEX


def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 90, 24)
    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=2)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    return img_erode


def find_tip(points, convex_hull):
    length = len(points)
    indices = np.setdiff1d(range(length), convex_hull)

    for i in range(2):
        j = indices[i] + 2
        if j > length - 1:
            j = length - j
        if np.all(points[j] == points[indices[i - 1] - 2]):
            return tuple(points[j])


def read_img_from_arrow_contours(img):
    result_side_cam = False
    # path_data_read_dir = file_path_data_read["arrow_dir_data_path"]
    # path_data_read_pos = file_path_data_read["arrow_pos_data_path"]

    contours, hierarchy = cv2.findContours(preprocess(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(img, contours, -1, 255, 3)

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        hull = cv2.convexHull(approx, returnPoints=False)
        sides = len(hull)

        if 6 > sides > 3 and sides + 2 == len(approx):
            arrow_tip = find_tip(approx[:, 0, :], hull.squeeze())

            print("++++++++++++++++++++++++++++++++++++++++++++++++")
            if arrow_tip:
                print("Calculating Direction of Arrow ")

                print("Sides", sides)

                cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)
                cv2.circle(img, arrow_tip, 10, (0, 0, 255), cv2.FILLED)

                ####### Calculate center of image ######

                # need center of contour
                M = cv2.moments(cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                cv2.circle(img, [cX, cY], 10, (255, 255, 0), cv2.FILLED)

                center = [cX, cY]

                difference_of_x = abs(arrow_tip[0] - center[0])
                difference_of_y = abs(arrow_tip[1] - center[1])

                if difference_of_x < 10 and difference_of_y > 50:

                    # Up and Down
                    if arrow_tip[1] < center[1]:
                        result_side_cam = "Up"
                        # write_data_for_dir(data="up", file_path=path_data_read_dir)
                        # write_data_for_pos(data="side", file_path=path_data_read_pos)
                        cv2.putText(img, 'UP', (20, 30), font, 1, (125, 125, 255), 2)

                    else:
                        result_side_cam = "Down"

                        # write_data_for_dir(data="down", file_path=path_data_read_dir)
                        # write_data_for_pos(data="side", file_path=path_data_read_pos)
                        cv2.putText(img, 'DOWN', (20, 30), font, 1, (125, 125, 255), 2)


                else:
                    # Right and Left
                    if arrow_tip[0] < center[0]:
                        result_side_cam = "Left"

                        # write_data_for_dir(data="left", file_path=path_data_read_dir)
                        # write_data_for_pos(data="side", file_path=path_data_read_pos)
                        cv2.putText(img, 'LEFT', (20, 30), font, 1, (125, 125, 255), 2)
                    else:
                        result_side_cam = "Right"

                        # write_data_for_dir(data="right", file_path=path_data_read_dir)
                        # write_data_for_pos(data="side", file_path=path_data_read_pos)
                        cv2.putText(img, 'RIGHT', (20, 30), font, 1, (125, 125, 255), 2)

                break

    # cv2.imshow("arrow_detect_with_contours", img)
    # cv2.waitKey(0)

    return result_side_cam


def read_img_from_arrow_contours_top(img, tmp_img, num_box):
    # path_data_read_dir = file_path_read_data["arrow_dir_data_path"]
    # path_data_read_pos = file_path_read_data["arrow_pos_data_path"]

    result_from_top_arrow_detect = False

    contours, hierarchy = cv2.findContours(preprocess(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    sp_case = False
    arrow_exist = False
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        hull = cv2.convexHull(approx, returnPoints=False)
        sides = len(hull)

        if cv2.contourArea(cnt) > 300:
            if cv2.contourArea(cnt) > 10000:
                # print("area.......:", cv2.contourArea(cnt))

                cv2.drawContours(img, [cnt], -1, (0, 0, 0), 3)
                # print("sides:", len(hull))
                # print("approx: ", len(approx))

                try:
                    arrow_tip = find_tip(approx[:, 0, :], hull.squeeze())

                except:
                    sp_case = True

            # if sp_case:
            #     M = cv2.moments(cnt)
            #     cX = int(M["m10"] / M["m00"])
            #     cY = int(M["m01"] / M["m00"])
            #
            #
            #     cv2.circle(img, [cX, cY], 10, (0, 0, 0), cv2.FILLED)
            #
            #
            #
            #
            #     approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
            #
            #     # draws boundary of contours.
            #     cv2.drawContours(img, [approx], 0, (0, 0, 255), 5)
            #     n = approx.ravel()
            #     i = 0
            #     arrow_tip = None
            #     for j in n:
            #         if (i % 2 == 0):
            #             x = n[i]
            #             y = n[i + 1]
            #
            #             if (i == 0):
            #
            #                 arrow_tip = [x,y]
            #
            #         i = i + 1
            #
            #     if arrow_tip is not None:
            #         cv2.circle(img, arrow_tip, 10, (0, 0, 0), cv2.FILLED)
            #
            #         h, w, c = img.shape
            #         p1 = [w,cY]
            #         cv2.circle(img, p1, 10, (255, 0, 0), cv2.FILLED)
            #
            #         p2 = [cX,cY]
            #         p3 = arrow_tip
            #
            #         try:
            #             # draw line to visulize
            #             cv2.line(img, p2, p1, (0, 0, 255), 2)
            #             cv2.line(img, p2, p3, (0, 0, 255), 2)
            #         except:
            #             pass
            #
            #         x1, y1 = p1
            #         x2, y2 = p2
            #         x3, y3 = p3
            #         deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
            #         deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
            #         angle =  int(deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2))
            #
            #         if angle > 315 or angle< 45:
            #             side = "Right "
            #             arrow_exist = True
            #             print("++++++++++++++++++++++++++++++++++++")
            #             print("area.......:", cv2.contourArea(cnt))
            #             print("++++++++++++++++++++++++++++++++++++")
            #
            #
            #         elif 135>angle > 45:
            #             side = "Up"
            #             arrow_exist = True
            #
            #             print("++++++++++++++++++++++++++++++++++++")
            #             print("area.......:", cv2.contourArea(cnt))
            #             print("++++++++++++++++++++++++++++++++++++")
            #
            #
            #         elif 225>angle > 135:
            #             side = "left"
            #             arrow_exist = True
            #
            #
            #         elif 315>angle > 225:
            #             side = "Down"
            #             arrow_exist = True
            #
            #
            #         else:
            #             side = None
            #             arrow_exist = False
            #
            #
            #
            #         if side is not  None:
            #             label = "Ori : " + str(angle)
            #
            #             cv2.putText(img, label + "/"+ side, (20, 30),
            #                         cv2.FONT_HERSHEY_SIMPLEX, 1, (24, 24, 228), 4, cv2.LINE_AA)
            #     #
            #
            #     break

            if 6 > sides > 3 and sides + 2 == len(approx) and not sp_case:
                arrow_tip = find_tip(approx[:, 0, :], hull.squeeze())

                # print("++++++++++++++++++++++++++++++++++++++++++++++++")
                if arrow_tip:

                    cv2.drawContours(tmp_img, [cnt], -1, (0, 255, 0), 3)
                    cv2.circle(tmp_img, arrow_tip, 10, (0, 0, 255), cv2.FILLED)

                    ####### Calculate center of image ######

                    # need center of contour
                    M = cv2.moments(cnt)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    cv2.circle(tmp_img, [cX, cY], 10, (255, 255, 0), cv2.FILLED)

                    center = [cX, cY]

                    # check distance between tip and center

                    if __get_distance_between_two_points(center, arrow_tip) > 70:
                        h, w, c = img.shape
                        p1 = [w, cY]
                        cv2.circle(img, p1, 10, (255, 0, 0), cv2.FILLED)

                        p2 = center
                        p3 = arrow_tip

                        try:
                            # draw line to visulize
                            cv2.line(img, p2, p1, (0, 0, 255), 2)
                            cv2.line(img, p2, p3, (0, 0, 255), 2)
                        except:
                            pass

                        x1, y1 = p1
                        x2, y2 = p2
                        x3, y3 = p3
                        deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
                        deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
                        angle = int(deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2))

                        if angle > 315 or angle < 45:
                            side = "Right"
                            arrow_exist = True
                            result_from_top_arrow_detect = "Right"
                            # write_data_for_dir(data="right", file_path=path_data_read_dir)
                            # write_data_for_pos(data="top", file_path=path_data_read_pos)


                        elif 135 > angle > 45:
                            side = "Up"
                            arrow_exist = True
                            result_from_top_arrow_detect = "Up"

                            # write_data_for_dir(data="up", file_path=path_data_read_dir)
                            # write_data_for_pos(data="top", file_path=path_data_read_pos)


                        elif 225 > angle > 135:
                            side = "left"
                            arrow_exist = True
                            result_from_top_arrow_detect = "left"

                            # write_data_for_dir(data="left", file_path=path_data_read_dir)
                            # write_data_for_pos(data="top", file_path=path_data_read_pos)


                        elif 315 > angle > 225:
                            side = "Down"
                            arrow_exist = True
                            result_from_top_arrow_detect = "Down"

                            # write_data_for_dir(data="down", file_path=path_data_read_dir)
                            # write_data_for_pos(data="top", file_path=path_data_read_pos)


                        else:
                            side = None
                            arrow_exist = False

                        if side is not None:
                            label = "Ori : " + str(angle)
                            #
                            # cv2.putText(img, label + "/" + side, (20, 30),
                            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (24, 24, 228), 4, cv2.LINE_AA)

                        break


        else:
            arrow_exist = False
    # print("arrow exist:", arrow_exist)
    # cv2.imshow("arrow_detect_with_contours"+str(num_box), img)
    # cv2.waitKey(0)

    return result_from_top_arrow_detect


def get_arrow(image, x1y1=None, x2y2=None):
    result = False
    try:
        # crop = image[x1y1[1] - 10:x2y2[1] + 10, x1y1[0] - 30:x2y2[0] + 30]

        crop = image[x1y1[1]:x2y2[1], x1y1[0]:x2y2[0]]
        # x_11 = int((x2y2[0]-x1y1[0])/2)
        # # y_11 = int((x2y2[1]-x1y1[1])/2)
        # y_11 = x1y1[1]
        # # #
        # crop_middle = crop[y_11:x2y2[1], x_11-10:x2y2[0]]
        # Now here we will detect direction arrow
        # __calculate_direction_of_arrow_according_1122(crop_middle)
        result = read_img_from_arrow_contours(crop)
        # cv2.imshow("Crop", image)
        # cv2.imshow("Crop_middle", crop_middle)
    except:
        print("No crop image")

        # crop = image[x1y1[1]:x2y2[1], x1y1[0]:x2y2[0]]
        result = read_img_from_arrow_contours(image)
    return result


def get_arrow_on_top(image, tmp_img, num_box):
    result = read_img_from_arrow_contours_top(image, tmp_img, num_box)

    return result
