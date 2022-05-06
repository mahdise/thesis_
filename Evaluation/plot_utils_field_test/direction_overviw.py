import math

import matplotlib.pyplot as plt

from Evaluation.utils import read_data, label_transform_direction_back

data_frame_1111 = read_data(
    "/Evaluation/processed_data_field_test/preprocessed_gt_0011.csv")
data_frame_1122 = read_data(
    "/Evaluation/processed_data_field_test/preprocessed_gt_0012.csv")
data_frame_1133 = read_data(
    "/Evaluation/processed_data_field_test/preprocessed_gt_0013.csv")
data_frame_2210 = read_data(
    "/Evaluation/processed_data_field_test/preprocessed_gt_0021.csv")
data_frame_2211 = read_data(
    "/Evaluation/processed_data_field_test/preprocessed_gt_0022.csv")
data_frame_2212 = read_data(
    "/Evaluation/processed_data_field_test/preprocessed_gt_0023.csv")
data_frame_3311 = read_data(
    "/Evaluation/processed_data_field_test/preprocessed_gt_0024.csv")
data_frame_3312 = read_data(
    "/Evaluation/processed_data_field_test/preprocessed_gt_0031.csv")
data_frame_4411 = read_data(
    "/Evaluation/processed_data_field_test/preprocessed_gt_0032.csv")
data_frame_4412 = read_data(
    "/Evaluation/processed_data_field_test/preprocessed_gt_0033.csv")

all_data_frame = [data_frame_1111, data_frame_1122, data_frame_1133,
                  data_frame_2210, data_frame_2211, data_frame_2211,
                  data_frame_3311, data_frame_3312, data_frame_4411,
                  data_frame_4412, ]

column_name = "arrow_direction"
actual_value_column_name = "arrow_direction_gt_1"

list_of_value = list()
list_of_experiment_name = list()
list_actual_value = list()
std_pred_value_list = list()

experiment_name = ["Vertical", "Horizental ", "Z-Horizental ",
                   "Short distance ", " Medium distance",
                   "Long distance",
                   "Attach box", "Small size ",
                   "Medium size ", " Large size "]

for index, data in enumerate(all_data_frame):
    # data = data.fillna(method='ffill')
    # data = data.fillna(method='bfill')
    data = label_transform_direction_back(data, actual_value_column_name)
    list_data_pred = list(data[column_name])
    list_data_actual = list(data[actual_value_column_name + "_encode"])
    # print(list_data_pred)
    if list_data_actual[0] is None:
        list_data_actual = ["None"] * len(list_data_actual)

    for index_, y in enumerate(list_data_pred):
        try:
            if math.isnan(y):
                list_data_pred[index_] = "None"
        except:
            pass

    x = range(0, len(data))

    fig, ax = plt.subplots()
    rects1 = ax.plot(x, list_data_pred, "d", label='Detect', color="lightblue")
    # if ok_ is not None:
    rects3 = ax.plot(x, list_data_actual, label='Actual', color="#ff8c00")

    ax.legend()

    name_of_x_axis = experiment_name[index] + " test"

    plt.xlabel(name_of_x_axis)
    plt.ylabel("Direction")

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)
    fig.tight_layout()

    legend = plt.legend(loc='upper left', edgecolor="black")
    legend.get_frame().set_alpha(0.1)
    # plt.rcParams["font.family"] = "Lato"
    # fig.set_size_inches(18.5, 10.5)

    save_name = name_of_x_axis + ".png"
    plt.savefig(save_name)

    plt.show()
