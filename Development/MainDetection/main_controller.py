"""
This script will be used for the connection with
other features like :
1. Roi selection
2. Arrow detection
3. Orientation detection


Step 01:
get multi cam connection
Step 02:
After got two image - load model
Step 03:
then prepare output

"""
import pathlib
import time
from multiprocessing import Process, Queue

import cv2
import numpy as np
import tensorflow as tf

# from Development.ExtractRoi.select_roi_by_mouse import click_and_crop
from Development.MainDetection.data_read_write import check_file_exist, create_csv_file, \
    __orientation_history
from Development.MainDetection.final_decession import final_decision
from Development.PipelineCamera.real_sense_camera_frame import PipelineCamera
from Development.yolov3_tf2.dataset import transform_images
# from Development.YoloModelTrained.models import YoloV3
# from Development.YoloModelTrained.datasets import transform_images
# from Development.YoloModelTrained.model_output import draw_outputs
from Development.yolov3_tf2.models import YoloV3
from Development.yolov3_tf2.utils import draw_outputs_top_cam, draw_outputs_side_cam
from camera_serial_num import __get_serial_num

refPt = []
cropping = False
crop = None


def resize_image(img, size, padColor=0):
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA
    else:  # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w / h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1:  # horizontal image
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:  # vertical image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:  # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor,
                                              (list, tuple, np.ndarray)):  # color image but only one color provided
        padColor = [padColor] * 3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT,
                                    value=padColor)

    return scaled_img


def crop_square(img, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h, w])

    # Centralize and crop
    crop_img = img[int(h / 2 - min_size / 2):int(h / 2 + min_size / 2),
               int(w / 2 - min_size / 2):int(w / 2 + min_size / 2)]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)

    return resized


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False


def side_camera_pipeline(operation_name, serial_num_of_cam, classes_name, data_from_side_cam: Queue):
    crop = None
    print(operation_name + "is starting............")
    main_object_get_pipeline = PipelineCamera(chosen_camera_serial_number=serial_num_of_cam)
    print("created pipeline")

    yolo = YoloV3()
    yolo.load_weights('../yolov3_tf2/weights/yolov3-custom.tf')
    print("Model loaded for " + operation_name)

    fps = 0.0

    while True:
        try:

            frames = main_object_get_pipeline.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())

            count = 0
            if color_image is None:
                time.sleep(0.1)
                count += 1
                if count < 10:
                    continue
                else:
                    break
            # ROI
            clone = color_image.copy()
            cv2.setMouseCallback(operation_name, click_and_crop)
            default = True
            if len(refPt) == 2:
                if refPt[0] != refPt[1]:
                    y = refPt[0][1]
                    x = refPt[0][0]
                    y_h = refPt[1][1]
                    x_w = refPt[1][0]
                    crop = clone[y:y_h, x:x_w]
                    cv2.rectangle(color_image, refPt[0], refPt[1], (36, 255, 12), 2)
                    default = False
                else:
                    default = True

            if default:
                y = 30
                x = 30
                h = 250
                w = 300

                crop = clone[y:y + h, x:x + w]
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (36, 255, 12), 2)
            # crop = color_image

            # Further operation will into only ROI
            if crop is not None:
                img_in = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                img_in = tf.expand_dims(img_in, 0)
                img_in = transform_images(img_in, 416)

                t1 = time.time()
                boxes, scores, classes, nums = yolo.predict(img_in)
                fps = (fps + (1. / (time.time() - t1))) / 2
                # # extract the bounding box coordinates
                img = draw_outputs_side_cam(crop, (boxes, scores, classes, nums), classes_name,
                                            data_from_side_cam)

            cv2.imshow(operation_name, color_image)
            # cv2.imshow(operation_name + "_output", img)
            if cv2.waitKey(1) == ord('q'):
                break
        except Exception as e:
            print(e)


def top_camera_pipeline(operation_name, serial_num_of_cam, classes_name,
                        data_from_top_cam: Queue):
    crop = None
    print(operation_name + "is starting............")
    main_object_get_pipeline = PipelineCamera(chosen_camera_serial_number=serial_num_of_cam)
    print("created pipeline")

    yolo = YoloV3()
    yolo.load_weights('../yolov3_tf2/weights/yolov3-custom.tf')
    print("Model loaded for " + operation_name)

    fps = 0.0

    while True:

        frames = main_object_get_pipeline.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        count = 0
        if color_image is None:
            time.sleep(0.1)
            count += 1
            if count < 10:
                continue
            else:
                break
        # ROI
        clone = color_image.copy()
        cv2.setMouseCallback(operation_name, click_and_crop)
        default = True
        if len(refPt) == 2:
            if refPt[0] != refPt[1]:
                y = refPt[0][1]
                x = refPt[0][0]
                y_h = refPt[1][1]
                x_w = refPt[1][0]
                crop = clone[y:y_h, x:x_w]
                cv2.rectangle(color_image, refPt[0], refPt[1], (36, 255, 12), 2)
                default = False
            else:
                default = True

        if default:
            y = 30
            x = 30
            h = 250
            w = 300

            crop = clone[y:y + h, x:x + w]
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (36, 255, 12), 2)

        # Further operation will into only ROI
        if crop is not None:
            img_in = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img_in = tf.expand_dims(img_in, 0)
            img_in = transform_images(img_in, 416)

            t1 = time.time()
            boxes, scores, classes, nums = yolo.predict(img_in)
            fps = (fps + (1. / (time.time() - t1))) / 2
            # # extract the bounding box coordinates
            img = draw_outputs_top_cam(crop, (boxes, scores, classes, nums), classes_name,
                                       data_from_top_cam)

        cv2.imshow(operation_name, color_image)
        cv2.imshow(operation_name + "_output", img)
        if cv2.waitKey(1) == ord('q'):
            break


def check_and_create_csv_file(need_check=True):
    if need_check:
        get_directory_path = str(pathlib.Path(__file__).parent.resolve())
        all_data_read_path = {
            "orientation_result": get_directory_path + "/orientation_result.csv",
        }
        all_data_read_path_exist_result = dict()
        for name, __check_file_exist in all_data_read_path.items():
            result_check_file_exist = check_file_exist(__check_file_exist)
            if not result_check_file_exist:
                all_data_read_path_exist_result[name] = False

        if len(all_data_read_path_exist_result) != 0:
            for name, path in all_data_read_path_exist_result.items():
                full_path = all_data_read_path[name]
                # create file
                if name == "orientation_result":
                    create_csv_file(header_name=__orientation_history, csv_name=full_path)

        return get_directory_path, all_data_read_path


def check_camera():
    serial_num_of_connected_camera = __get_serial_num()
    if len(serial_num_of_connected_camera) != 0:
        if len(serial_num_of_connected_camera) == 2:
            return serial_num_of_connected_camera
        elif len(serial_num_of_connected_camera) < 2:
            print("Connected Camera Less Then Two")
            return serial_num_of_connected_camera

        else:
            print("Connected RealSense Camera More Then Two. We Will Take First Two")
            return serial_num_of_connected_camera
    else:
        return None


if __name__ == '__main__':

    # get connected camera list number
    connected_cameras = check_camera()

    if connected_cameras is not None:

        # Check device
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        # load class name for model
        class_names = [c.strip() for c in open('../yolov3_tf2/labels/obj.names').readlines()]

        # check data read file exist or not
        file_path, data_read_path = check_and_create_csv_file()

        # For put the detection data for analysis by final decision pipeline
        top_cam_data = Queue()
        side_cam_data = Queue()

        # create multiprocess
        top_cam_process = Process(target=top_camera_pipeline,
                                  args=('Rotate_Angle_Detect_ROI', connected_cameras[0], class_names, top_cam_data))
        side_cam_process = Process(target=side_camera_pipeline,
                                   args=(
                                       'Arrow_Detect_ROI', connected_cameras[1], class_names, side_cam_data))
        data_read_process = Process(target=final_decision,
                                    args=(top_cam_data, side_cam_data, data_read_path))

        # add process to list for start and join
        all_process = [top_cam_process, side_cam_process, data_read_process]
        for _process in all_process:
            _process.start()
        for _pJoin in all_process:
            _pJoin.join()
