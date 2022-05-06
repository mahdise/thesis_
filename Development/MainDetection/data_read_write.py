import csv
import os

import pandas as pd

__default_csv_name_pos = "detection_info_arrow_pos.csv"
__header_name_pos = ["arrow_position"]
__header_name_dir = ["arrow_direction"]
__header_name_angle = ["box_number", "rotate_angle", "arrow_position", "arrow_direction", "orientation_type"]
__default_csv_name_dir = "detection_info_arrow_dir.csv"
__default_csv_name_angle = "detection_info_angle.csv"

# for evaluated only
__orientation_history = ["box_number", "rotate_angle", "arrow_position", "arrow_direction",
                         "orientation_type", "box_detect_confidence_top_cam", "box_detect_confidence_side_cam",
                         "total_boxes", "angle_type", "data_write_time"]


def check_file_exist(file_path):
    result = False
    if os.path.isfile(file_path):
        result = True

    return result


def create_csv_file(header_name=None, csv_name=None):
    result = False

    try:
        with open(csv_name, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(header_name)

            result = True
    except:
        result = False

    return result


def create_csv_file_position():
    create_pos_csv = create_csv_file(header_name=__header_name_pos,
                                     csv_name=__default_csv_name_pos)
    return create_pos_csv


def create_csv_file_dir():
    create_dir_csv = create_csv_file(header_name=__header_name_dir,
                                     csv_name=__default_csv_name_dir)
    return create_dir_csv


def create_csv_file_angle():
    create_angle_csv = create_csv_file(header_name=__header_name_angle,
                                       csv_name=__default_csv_name_angle)
    return create_angle_csv


def make_empty_csv_file(csv_path):
    empty_data = pd.DataFrame()
    empty_data.to_csv(csv_path, index=False, header=True)


def write_data_2nd_version(data_frame, csv_path):
    data_frame.to_csv(csv_path, index=False, header=True)


def write_data(data, csv_name):
    data_frame = pd.read_csv(csv_name)
    data_frame.loc[-1] = data
    data_frame.index = data_frame.index + 1  # shifting index
    data_frame = data_frame.sort_index()

    save_data = data_frame.to_csv(csv_name, index=False, header=True)

    return save_data


def write_data_for_pos(data, file_path=None):
    if file_path is None:
        csv_name = __default_csv_name_pos
    else:
        csv_name = file_path

    write_data(data=data, csv_name=csv_name)


def write_data_for_dir(data, file_path=None):
    if file_path is None:
        csv_name = __default_csv_name_dir
    else:
        csv_name = file_path
    write_data(data=data, csv_name=csv_name)


def write_data_for_angle(data, file_path=None):
    if file_path is None:
        csv_name = __default_csv_name_angle
    else:
        csv_name = file_path

    write_data(data=data, csv_name=csv_name)


def del_csv(csv_name):
    os.remove(csv_name)
