import pandas as pd
from sklearn.preprocessing import LabelEncoder


# temp_data = temp_data.replace({"Right": 2,
#                                "Left": 1,
#                                "Up": 3,
#                                "Down": 4, })

def read_data(data_path):
    return pd.read_csv(data_path)


def label_transform_position_back(data_frame_for_label_encode, column_name):
    temp_data = pd.DataFrame(data={
        "actual_data": data_frame_for_label_encode[column_name],
    })
    temp_data = temp_data.replace({1: "Top",
                                   2: "Side",
                                   0: None})
    data_frame_for_label_encode[column_name + "_encode"] = temp_data["actual_data"]

    return data_frame_for_label_encode


def label_transform_direction_back(data_frame_for_label_encode, column_name):
    temp_data = pd.DataFrame(data={
        "actual_data": data_frame_for_label_encode[column_name],
    })
    temp_data = temp_data.replace({2: "Right",
                                   1: "Left",
                                   3: "Up",
                                   4: "Down",
                                   0: None})

    data_frame_for_label_encode[column_name + "_encode"] = temp_data["actual_data"]

    return data_frame_for_label_encode


def label_transform_position(data_frame_for_label_encode, column_name):
    temp_data = pd.DataFrame(data={
        "actual_data": data_frame_for_label_encode[column_name],
    })
    temp_data = temp_data.replace({"Top": 1,
                                   "Side": 2, })
    data_frame_for_label_encode[column_name + "_encode"] = temp_data["actual_data"]

    return data_frame_for_label_encode


def label_transform_direction(data_frame_for_label_encode, column_name):
    temp_data = pd.DataFrame(data={
        "actual_data": data_frame_for_label_encode[column_name],
    })
    temp_data = temp_data.replace({"Right": 2,
                                   "Left": 1,
                                   "Up": 3,
                                   "Down": 4, })

    data_frame_for_label_encode[column_name + "_encode"] = temp_data["actual_data"]

    return data_frame_for_label_encode


def label_transform_angle_type(data_frame_for_label_encode, column_name):
    temp_data = pd.DataFrame(data={
        "actual_data": data_frame_for_label_encode[column_name],
    })
    temp_data = temp_data.replace({"Right": 2,
                                   "Straight": 3,
                                   "Obtuse": 1,
                                   "Acute": 4, })

    data_frame_for_label_encode[column_name + "_encode"] = temp_data["actual_data"]

    return data_frame_for_label_encode


def label_encode(data_frame_for_label_encode, column_name):
    # creating instance of label encoder
    label_encoder = LabelEncoder()
    # Assigning numerical values and storing in another column
    data_frame_for_label_encode[column_name + "_encode"] = label_encoder.fit_transform(
        data_frame_for_label_encode[column_name])

    return data_frame_for_label_encode


def column_to_numeric(data_frame_for_change_type, column_name, change_type="float", sp_word=None):
    if sp_word:
        data_frame_for_change_type[column_name] = data_frame_for_change_type[column_name].str.rstrip(sp_word)

    data_frame_for_change_type[column_name] = data_frame_for_change_type[column_name].astype(change_type)

    return data_frame_for_change_type
