import pandas as pd

from Evaluation.utils import read_data

data_frame_1111 = read_data(
    "/Evaluation/statistical_data_field_test/statistical_1_preprocessed_0011.csv")
data_frame_1122 = read_data(
    "/Evaluation/statistical_data_field_test/statistical_1_preprocessed_0012.csv")
data_frame_1133 = read_data(
    "/Evaluation/statistical_data_field_test/statistical_1_preprocessed_0013.csv")
data_frame_2210 = read_data(
    "/Evaluation/statistical_data_field_test/statistical_1_preprocessed_0021.csv")
data_frame_2211 = read_data(
    "/Evaluation/statistical_data_field_test/statistical_2_preprocessed_0021.csv")
data_frame_2212 = read_data(
    "/Evaluation/statistical_data_field_test/statistical_1_preprocessed_0022.csv")
data_frame_3311 = read_data(
    "/Evaluation/statistical_data_field_test/statistical_2_preprocessed_0022.csv")
data_frame_3312 = read_data(
    "/Evaluation/statistical_data_field_test/statistical_1_preprocessed_0023.csv")
data_frame_4411 = read_data(
    "/Evaluation/statistical_data_field_test/statistical_2_preprocessed_0023.csv")
data_frame_4412 = read_data(
    "/Evaluation/statistical_data_field_test/statistical_1_preprocessed_0024.csv")
data_frame_5511 = read_data(
    "/Evaluation/statistical_data_field_test/statistical_2_preprocessed_0024.csv")
data_frame_5512 = read_data(
    "/Evaluation/statistical_data_field_test/statistical_1_preprocessed_0031.csv")
data_frame_5513 = read_data(
    "/Evaluation/statistical_data_field_test/statistical_1_preprocessed_0032.csv")
data_frame_6611 = read_data(
    "/Evaluation/statistical_data_field_test/statistical_1_preprocessed_0033.csv")

all_data_frame = [data_frame_1111, data_frame_1122, data_frame_1133,
                  data_frame_2210, data_frame_2211, data_frame_2211,
                  data_frame_3311, data_frame_3312, data_frame_4411,
                  data_frame_4412, data_frame_5511, data_frame_5512,
                  data_frame_5513, data_frame_6611, ]

data_frame_experiment_label = ["Vertical", "Horizental ", "Z-Horizental ",
                               "Short distance box 1", " Short distance box 2", " Medium distance box 1",
                               "Medium distance box 2", "Long distance box 1", "Long distance box 2",
                               "Attach box 1", " Attach box 2", "Small size ",
                               "Medium size ", " Large size "]
# parameter name ex. case of mape ( mean average percentage error )
parameter_name = 12
# column_name = "rotate_angle"
# column_name = "arrow_position_encode"
# column_name = "arrow_direction_encode"
# column_name = "box_detect_confidence_top_cam"
# column_name = "box_detect_confidence_side_cam"
column_name = "rotate_angle"

result = dict()

list_value_mape = list()
list_value_count = list()
list_value_mean = list()
list_value_std = list()
list_value_priccision = list()
list_value_recall = list()
list_value_rmse = list()
list_value_accuracy = list()
for index_num, data_st in enumerate(all_data_frame):

    for column in data_st:

        if column == column_name:
            all_value = data_st[column]
            for index, value in enumerate(all_value):
                if index == 0:
                    list_value_count.append(value)
                elif index == 1:
                    list_value_mean.append(value)

                elif index == 2:
                    list_value_std.append(value)

                elif index == 8:
                    list_value_priccision.append(value)

                elif index == 9:
                    list_value_recall.append(value)


                elif index == 10:
                    list_value_rmse.append(value)

                elif index == 12:
                    list_value_mape.append(value)

                elif index == 13:
                    list_value_accuracy.append(value)
        else:
            continue

result = {
    "name_of_experiment": data_frame_experiment_label,
    "accuracy": list_value_accuracy,
    "mape": list_value_mape,
    "rmse": list_value_rmse,
    "reacall": list_value_recall,
    "priccision": list_value_priccision,
    "std": list_value_std,
    "mean": list_value_mean,
    "count": list_value_count,
    "angle_gt": [82,
                 86,
                 125,
                 102,
                 139,
                 119,
                 111,
                 117,
                 129,
                 115,
                 140,
                 116,
                 139,
                 61]

}

result_to_data_frame = pd.DataFrame.from_dict(result)
# csv_name = "latest_experiment_side_cam.csv"
# result_to_data_frame.to_csv(
#     r'/home/mahdiislam/Mahdi/Biba/iris-parcel-orientation-detection/Evalouation/statistical_data_field_test/' + csv_name,
#     index=True, header=True)

print(result_to_data_frame)
