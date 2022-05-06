from Evaluation.utils import read_data, label_encode,\
    column_to_numeric, label_transform_position, label_transform_direction,label_transform_angle_type
import pandas as pd


def preprocess_data(data_path, save_data=True):
    data_frame = read_data(data_path)

    csv_name = "preprocessed_" + data_path.rpartition('/')[2]

    # Change type of column
    confidence_box_detection_type_top_cam = column_to_numeric(
        data_frame_for_change_type=data_frame,
        column_name="box_detect_confidence_top_cam",
        sp_word="%",
        change_type="float"
    )

    confidence_box_detection_type_side_cam = column_to_numeric(
        data_frame_for_change_type=confidence_box_detection_type_top_cam,
        column_name="box_detect_confidence_side_cam",
        sp_word="%",
        change_type="float"
    )

    # label encode
    encode_arrow_position = label_transform_position(data_frame_for_label_encode=confidence_box_detection_type_side_cam,
                                         column_name="arrow_position")
    encode_arrow_direction = label_transform_direction(data_frame_for_label_encode=encode_arrow_position,
                                          column_name="arrow_direction")

    encode_orientation_type = label_encode(data_frame_for_label_encode=encode_arrow_direction,
                                           column_name="orientation_type")

    encode_angle_type = label_transform_angle_type(data_frame_for_label_encode=encode_orientation_type, column_name="angle_type")

    final_data_frame = encode_angle_type

    # save data

    if save_data:
        final_data_frame.to_csv(
            r'/home/mahdiislam/Mahdi/Biba/iris-parcel-orientation-detection/Evalouation/ProcessedData/' + csv_name,
            index=False, header=True)

def preprocess_data_with_gt(data_path, save_data=True):
    data_frame = read_data(data_path)

    csv_name = "preprocessed_gt_" + data_path.rpartition('/')[2]

    # Change type of column
    confidence_box_detection_type_top_cam = column_to_numeric(
        data_frame_for_change_type=data_frame,
        column_name="box_detect_confidence_top_cam",
        sp_word="%",
        change_type="float"
    )

    confidence_box_detection_type_side_cam = column_to_numeric(
        data_frame_for_change_type=confidence_box_detection_type_top_cam,
        column_name="box_detect_confidence_side_cam",
        sp_word="%",
        change_type="float"
    )

    # label encode
    encode_arrow_position = label_transform_position(data_frame_for_label_encode=confidence_box_detection_type_side_cam,
                                         column_name="arrow_position")
    encode_arrow_direction = label_transform_direction(data_frame_for_label_encode=encode_arrow_position,
                                          column_name="arrow_direction")

    encode_orientation_type = label_encode(data_frame_for_label_encode=encode_arrow_direction,
                                           column_name="orientation_type")

    encode_angle_type = label_transform_angle_type(data_frame_for_label_encode=encode_orientation_type, column_name="angle_type")

    encode_angle_type["angle_gt_1"] = [119]*len(encode_angle_type)
    encode_angle_type["box_number_gt_1"] = [1]*len(encode_angle_type)
    encode_angle_type["arrow_direction_gt_1"] = [1]*len(encode_angle_type)
    encode_angle_type["arrow_position_gt_1"] = [2]*len(encode_angle_type)
    encode_angle_type["orientation_type_gt_1"] = [21]*len(encode_angle_type)
    encode_angle_type["angle_type_gt_1"] = [1]*len(encode_angle_type)
    #
    # encode_angle_type["angle_gt_2"] = [73]*len(encode_angle_type)
    # encode_angle_type["box_number_gt_2"] = [2]*len(encode_angle_type)
    # encode_angle_type["arrow_direction_gt_2"] = [0]*len(encode_angle_type)
    # encode_angle_type["arrow_position_gt_2"] = [0]*len(encode_angle_type)
    # encode_angle_type["orientation_type_gt_2"] = [0]*len(encode_angle_type)
    # encode_angle_type["angle_type_type_gt_2"] = [4]*len(encode_angle_type)
    # #
    # encode_angle_type["angle_gt_3"] = [129]*len(encode_angle_type)
    # encode_angle_type["box_number_gt_3"] = [3]*len(encode_angle_type)
    # encode_angle_type["arrow_direction_gt_3"] = [0]*len(encode_angle_type)
    # encode_angle_type["arrow_position_gt_3"] = [0]*len(encode_angle_type)
    # encode_angle_type["orientation_type_gt_3"] = [0]*len(encode_angle_type)
    # encode_angle_type["angle_type_type_gt_3"] = [1]*len(encode_angle_type)

    final_data_frame = encode_angle_type

    # save data

    if save_data:
        final_data_frame.to_csv(
            r'/home/mahdiislam/Mahdi/Biba/iris-parcel-orientation-detection/Evalouation/ProcessedData/' + csv_name,
            index=False, header=True)

if __name__ == '__main__':
    test_data = "/home/mahdiislam/Mahdi/Biba/iris-parcel-orientation-detection/DataTest/LabTest/DataSecondVersion/orientation_result_1111.csv"
    preprocess_data_with_gt(test_data)
