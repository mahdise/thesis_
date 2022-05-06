import cv2
import numpy as np
from Development.MainDetection.data_read_write import write_data_for_pos, write_data_for_dir
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


def read_img_from_arrow_contours(img, file_path_data_read):
    path_data_read_dir = file_path_data_read["arrow_dir_data_path"]
    path_data_read_pos = file_path_data_read["arrow_pos_data_path"]

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

                if difference_of_x < 6 and difference_of_y > 50:
                    # Up and Down
                    if arrow_tip[1] < center[1]:
                        write_data_for_dir(data="up", file_path=path_data_read_dir)
                        write_data_for_pos(data="side", file_path=path_data_read_pos)
                        cv2.putText(img, 'UP', (20, 30), font, 1, (125, 125, 255), 2)

                    else:
                        write_data_for_dir(data="down", file_path=path_data_read_dir)
                        write_data_for_pos(data="side", file_path=path_data_read_pos)
                        cv2.putText(img, 'DOWN', (20, 30), font, 1, (125, 125, 255), 2)


                else:
                    # Right and Left
                    if arrow_tip[0] < center[0]:
                        write_data_for_dir(data="left", file_path=path_data_read_dir)
                        write_data_for_pos(data="side", file_path=path_data_read_pos)
                        cv2.putText(img, 'LEFT', (20, 30), font, 1, (125, 125, 255), 2)
                    else:
                        write_data_for_dir(data="right", file_path=path_data_read_dir)
                        write_data_for_pos(data="side", file_path=path_data_read_pos)
                        cv2.putText(img, 'RIGHT', (20, 30), font, 1, (125, 125, 255), 2)

                break

    cv2.imshow("arrow_detect_with_contours", img)
    # cv2.waitKey(0)


def get_arrow(image, x1y1=None, x2y2=None, file_path_data_read = None):
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
        read_img_from_arrow_contours(crop, file_path_data_read)
        # cv2.imshow("Crop", image)
        # cv2.imshow("Crop_middle", crop_middle)
    except:
        print("No crop image")

        # crop = image[x1y1[1]:x2y2[1], x1y1[0]:x2y2[0]]
        read_img_from_arrow_contours(image, file_path_data_read)

# if __name__ == '__main__':
#     image = cv2.imread("/home/mahdiislam/Mahdi/Biba/others_repo/Object-Detection-API/detect_arrow_dir/data/img_7.png")
#     get_arrow(image)
