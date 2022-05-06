from absl import logging
import numpy as np
import tensorflow as tf
import cv2
from seaborn import color_palette
from PIL import Image, ImageDraw, ImageFont
# import the necessary packages
from math import atan2, cos, sin, sqrt, pi
# from detect_arrow_dir.detect_arrow_1122 import __calculate_direction_of_arrow_according_1122
# from Development.DetectOrientation.main_controller_detect_orientation import __get_angle
from Development.DetectOrientation.orientation_detection_pipeline_final import __get_angle
# from Development.DetectArrow.main_controller_detect_arrow import get_arrow
from Development.DetectArrow.arrow_detection_pipeline_final import get_arrow

from datetime import datetime

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

YOLOV3_TINY_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
]


def load_darknet_weights(model, weights_file, tiny=False):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    if tiny:
        layers = YOLOV3_TINY_LAYER_LIST
    else:
        layers = YOLOV3_LAYER_LIST

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            logging.info("{}/{} {}".format(
                sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.input_shape[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
                 (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
                 (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def draw_outputs_side_cam(img, outputs, class_names, put_data_for_side_cam):
    # make empty data csv
    data_read_side_cam = dict()
    confidence = None
    box_detect_ = False
    copy_image = img
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font='../yolov3_tf2/fonts/futur.ttf',
                              size=(img.size[0] + img.size[1]) // 100)

    for i in range(nums):
        color = colors[int(classes[i])]
        x1y1 = ((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = ((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        thickness = (img.size[0] + img.size[1]) // 200
        x0, y0 = x1y1[0], x1y1[1]
        for t in np.linspace(0, 1, thickness):
            x1y1[0], x1y1[1] = x1y1[0] - t, x1y1[1] - t
            x2y2[0], x2y2[1] = x2y2[0] - t, x2y2[1] - t
            draw.rectangle([x1y1[0], x1y1[1], x2y2[0], x2y2[1]], outline=tuple(color))

        confidence = '{:.2f}%'.format(objectness[i] * 100)
        text = '{} {}'.format(class_names[int(classes[i])], confidence)
        text_size = draw.textsize(text, font=font)
        draw.rectangle([x0, y0 - text_size[1], x0 + text_size[0], y0],
                       fill=tuple(color))
        draw.text((x0, y0 - text_size[1]), text, fill='black',
                  font=font)

        data_read_side_cam["box_" + str(i + 1) + "_arrowSide"] = False
        data_read_side_cam["box_" + str(i + 1) + "_arrowDirection"] = False
        data_read_side_cam["box_" + str(i + 1) + "_confidence"] = confidence

        result_data_side_cam = get_arrow(copy_image, x1y1, x2y2)
        if result_data_side_cam:
            data_read_side_cam["box_" + str(i + 1) + "_arrowSide"] = True
            data_read_side_cam["box_" + str(i + 1) + "_arrowDirection"] = result_data_side_cam



    if not box_detect_:
        data_read_side_cam["box_0" + "_confidence"] = confidence

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    data_read_side_cam["read_data_time "] = current_time
    data_read_side_cam["total_boxes"] = nums
    put_data_for_side_cam.put(data_read_side_cam)

    rgb_img = img.convert('RGB')
    img_np = np.asarray(rgb_img)

    img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    return img


def draw_outputs_top_cam(img, outputs, class_names, put_data_for_top_cam):
    # make empty data csv
    data_read_top_cam = dict()
    confidence = None
    box_detect_ = False

    copy_image = img
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font='../yolov3_tf2/fonts/futur.ttf',
                              size=(img.size[0] + img.size[1]) // 100)
    for i in range(nums):
        box_detect_ = True
        color = colors[int(classes[i])]
        x1y1 = ((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = ((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        thickness = (img.size[0] + img.size[1]) // 200
        x0, y0 = x1y1[0], x1y1[1]
        for t in np.linspace(0, 1, thickness):
            x1y1[0], x1y1[1] = x1y1[0] - t, x1y1[1] - t
            x2y2[0], x2y2[1] = x2y2[0] - t, x2y2[1] - t
            draw.rectangle([x1y1[0], x1y1[1], x2y2[0], x2y2[1]], outline=tuple(color))

        confidence = '{:.2f}%'.format(objectness[i] * 100)
        text = '{} {}'.format(class_names[int(classes[i])], confidence)
        text_size = draw.textsize(text, font=font)
        draw.rectangle([x0, y0 - text_size[1], x0 + text_size[0], y0],
                       fill=tuple(color))
        draw.text((x0, y0 - text_size[1]), text, fill='black',
                  font=font)

        result_data = __get_angle(copy_image, x1y1, x2y2, num_boxes=i + 1, x0=x0, y0=y0)
        data_read_top_cam["box_" + str(i + 1) + "_confidence"] = confidence

        if len(result_data) != 0:

            # put text of angel
            __angle_every_box = "Angle: " + str(result_data["angle"])
            draw.text((x0, y0 + 15 - text_size[1]), __angle_every_box, fill='green',
                      font=font)
            # draw center circle
            center_x, center_y = result_data["center"][0], result_data["center"][1]
            r = 10
            draw.ellipse([(center_x - r, center_y - r), (center_x + r, center_y + r)], fill='orange',
                         outline='black')

            # draw line
            draw.line(result_data["line_1"], fill="green", width=5)
            draw.line(result_data["line_2"], fill="green", width=5)
            draw.line(result_data["line_3"], fill="green", width=5)
            draw.line(result_data["line_4"], fill="green", width=5)
            draw.line(result_data["line_5"], fill="orange", width=5)
            draw.line(result_data["line_6"], fill="orange", width=5)

            # add data
            data_read_top_cam["box_" + str(i + 1) + "_angle"] = int(result_data["angle"])
            data_read_top_cam["box_" + str(i + 1) + "_angle_type"] = result_data["angle_type"]
            data_read_top_cam["box_" + str(i + 1) + "_arrowTop"] = False
            data_read_top_cam["box_" + str(i + 1) + "_arrowDirection"] = False

            if result_data["result_for_top_arrow"]:
                data_read_top_cam["box_" + str(i + 1) + "_arrowTop"] = True
                data_read_top_cam["box_" + str(i + 1) + "_arrowDirection"] = result_data["result_for_top_arrow"]

    if not box_detect_:
        data_read_top_cam["box_0" + "_confidence"] = confidence

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    data_read_top_cam["read_data_time "] = current_time
    data_read_top_cam["total_boxes"] = nums
    put_data_for_top_cam.put(data_read_top_cam)

    rgb_img = img.convert('RGB')
    img_np = np.asarray(rgb_img)

    img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)


    return img


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


def get_orinatation_0_180(c, img):
    # need to see
    # TODO : NEED TO SEE TO GET ANGLE
    # https://stackoverflow.com/questions/64860785/opencv-using-canny-and-shi-tomasi-to-detect-round-corners-of-a-playing-card

    # cv.minAreaRect returns:
    # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Retrieve the key parameters of the rotated bounding box
    center = (int(rect[0][0]), int(rect[0][1]))
    width = int(rect[1][0])
    height = int(rect[1][1])
    angle = int(rect[2])

    print("###########################################")
    print("angle-original : ", angle)

    if width < height:
        angle = 90 - angle
    else:
        angle = -angle
    print("After calculation ", angle)

    print("###########################################")

    # textbox = cv2.rectangle(img, (center[0] - 35, center[1] - 25),
    #                        (center[0] + 295, center[1] + 10), (255, 255, 255), -1)
    # cv2.putText(img, label, (center[0] - 50, center[1]),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)


# ref :
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
    print("After calculate angle", str(-int(np.rad2deg(angle)) - 90))
    # textbox = cv.rectangle(img, (cntr[0], cntr[1] - 25), (cntr[0] + 250, cntr[1] + 10), (255, 255, 255), -1)
    # cv.putText(img, label, (cntr[0], cntr[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

    return angle


# ref important : https://automaticaddison.com/how-to-determine-the-orientation-of-an-object-using-opencv/
# ref : https://medium.com/sicara/opencv-edge-detection-tutorial-7c3303f10788
# ref : https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
def auto_canny(image, sigma=0.33):
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


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

    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(crop, contours, -1, (0, 255, 0), 3)
    # im2, contours, hierarchy = cv2.findContours(thresh, 1, 1)
    print("############################################################")
    list_of_area = list()
    # for c in contours:
    #     # calculate area and moment of each contour
    #     area = cv2.contourArea(c)
    #
    #     # list of area
    #     list_of_area.append(area)
    #     M = cv2.moments(c)
    #
    #     if M["m00"] > 0:
    #         cX = int(M["m10"] / M["m00"])
    #         cY = int(M["m01"] / M["m00"])
    #
    #     # print("area : ",area)
    # max_area = max(list_of_area)
    # print(max_area)
    # print(max_area)
    for c in contours:
        # k = cv2.isContourConvex(c)
        # print(k)
        cv2.drawContours(crop, c, 0, (0, 255, 0), 3)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(crop, [box], 0, (0, 0, 255), 2)

    cv2.imwrite(
        "/home/mahdiislam/Mahdi/Biba/others_repo/Object-Detection-API/image_output_file/am-bounding-box.jpg",
        crop)

    # area = cv2.contourArea(c)

    # if area == max_area:
    #     rect = cv2.minAreaRect(c)
    #     box = cv2.boxPoints(rect)
    #     box = np.int0(box)
    #     cv2.drawContours(crop, [box], 0, (0, 0, 255), 2)
    #
    #     cv2.imwrite(
    #         "/home/mahdiislam/Mahdi/Biba/others_repo/Object-Detection-API/image_output_file/bounding-box.jpg",
    #         crop)
    # rect = cv2.minAreaRect(c)
    # box = cv2.boxPoints(rect)
    # print(box)
    # cv2.drawContours(crop, [box], 0, (182, 41, 41), 2)

    # for p in box:
    #     pt = (p[0], p[1])
    #     plt.scatter(p[0], p[1])

    print("############################################################")

    cv2.imshow("bounding-box", crop)
    cv2.imshow("bounding-box-gray", im_bw)

    # cv2.imshow("bounding-box-edge", edged)


def draw_labels(x, y, class_names):
    colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
    img = x.numpy()
    boxes, classes = tf.split(y, (4, 1), axis=-1)
    classes = classes[..., 0]
    wh = np.flip(img.shape[0:2])
    for i in range(len(boxes)):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, class_names[classes[i]],
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          1, (0, 0, 0), 2)
    return img


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)

# image = cv2.imread('/home/mahdiislam/Mahdi/Biba/others_repo/Object-Detection-API/savedImage.jpg')
#
# __get_angle_2(image)
