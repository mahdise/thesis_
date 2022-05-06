from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

list_None_count_pred = list()
list_right_count_pred = list()
list_left_count_pred = list()
list_up_count_pred = list()
list_down_count_pred = list()

list_None_count_actual = list()
list_right_count_actual = list()
list_left_count_actual = list()
list_up_count_actual = list()
list_down_count_actual = list()
for index, data in enumerate(all_data_frame):
    # data = data.fillna(method='ffill')
    # data = data.fillna(method='bfill')
    data = label_transform_direction_back(data, actual_value_column_name)

    list_data_pred = list(data[column_name])
    list_data_actual = list(data[actual_value_column_name + "_encode"])

    unqiue_items_number_pred = Counter(list_data_pred)
    unqiue_items_number_actual = Counter(list_data_actual)

    right_value_pred = unqiue_items_number_pred.get("Right") or 0
    left_value_pred = unqiue_items_number_pred.get("Left") or 0
    up_value_pred = unqiue_items_number_pred.get("Up") or 0
    down_value_pred = unqiue_items_number_pred.get("Down") or 0
    nan_value_pred = unqiue_items_number_pred.get(np.nan) or 0

    list_right_count_pred.append(right_value_pred)
    list_left_count_pred.append(left_value_pred)
    list_up_count_pred.append(up_value_pred)
    list_down_count_pred.append(down_value_pred)
    list_None_count_pred.append(nan_value_pred)

    right_value_actual = unqiue_items_number_actual.get("Right") or 0
    left_value_actual = unqiue_items_number_actual.get("Left") or 0
    up_value_actual = unqiue_items_number_actual.get("Up") or 0
    down_value_actual = unqiue_items_number_actual.get("Down") or 0
    nan_value_actual = unqiue_items_number_actual.get(np.nan) or 0

    list_right_count_actual.append(right_value_actual)
    list_left_count_actual.append(left_value_actual)
    list_up_count_actual.append(up_value_actual)
    list_down_count_actual.append(down_value_actual)
    list_None_count_actual.append(nan_value_actual)
# prepare data
final_data_dict = {
    "experiment_name": experiment_name,
    "right_pred": list_right_count_pred,
    "left_pred": list_left_count_pred,
    "up_pred": list_up_count_pred,
    "down_pred": list_down_count_pred,
    "none_pred": list_None_count_pred,

    "right_ac": list_right_count_actual,
    "left_ac": list_left_count_actual,
    "up_ac": list_up_count_actual,
    "down_ac": list_down_count_actual,
    "none_ac": list_None_count_actual,

}
result_to_data_frame = pd.DataFrame(data=final_data_dict)
print(result_to_data_frame)
# print(result_to_data_frame)
# csv_name = "pos_percenatage_of_good_bad_for_every_experiments.csv"
# result_to_data_frame.to_csv(
#     r'/home/mahdiislam/Mahdi/Biba/iris-parcel-orientation-detection/Evalouation/plot_data/' + csv_name,
#     index=True, header=True)


# plot functions


labels = result_to_data_frame["experiment_name"]

width = 0.3  # the width of the bars: can also be len(x) sequence
x = np.arange(len(labels))
fig, ax = plt.subplots()

rect_1 = [
    ax.bar(x - width / 2, final_data_dict["right_pred"], width, label='Predict Right', color="#008080"),
    ax.bar(x - width / 2, final_data_dict["left_pred"], width, bottom=final_data_dict["right_pred"],
           label='Predict Left', color="#32CD32"),

    ax.bar(x - width / 2, final_data_dict["up_pred"], width,
           bottom=np.array(final_data_dict["left_pred"]) + np.array(final_data_dict["right_pred"]),
           label='Predict Up', color="#ff8c00"),

    ax.bar(x - width / 2, final_data_dict["down_pred"], width,
           bottom=np.array(final_data_dict["left_pred"]) + np.array(final_data_dict["right_pred"]) + np.array(
               final_data_dict["up_pred"]),
           label='Predict Down', color="#c731d2"),
    #
    ax.bar(x - width / 2, final_data_dict["none_pred"], width,
           bottom=np.array(final_data_dict["left_pred"]) + np.array(final_data_dict["right_pred"]) + np.array(
               final_data_dict["up_pred"]) + np.array(final_data_dict["down_pred"]),
           label='Predict None', color="#983600"),
]

rect_2 = [
    ax.bar(x + width / 2, final_data_dict["right_ac"], width,
           color="#008080", edgecolor="black", label='actual Right'),
    ax.bar(x + width / 2, final_data_dict["left_ac"], width, bottom=final_data_dict["right_ac"],
           color="#32CD32", edgecolor="black", label='Actual Left'),

    ax.bar(x + width / 2, final_data_dict["up_ac"], width,
           bottom=np.array(final_data_dict["left_ac"]) + np.array(final_data_dict["right_ac"]),
           color="#ff8c00", edgecolor="black", label=' Actual Up'),

    ax.bar(x + width / 2, final_data_dict["down_ac"], width,
           bottom=np.array(final_data_dict["left_ac"]) + np.array(final_data_dict["right_ac"]) + np.array(
               final_data_dict["up_ac"]),
           color="#c731d2", edgecolor="black", label=' Actual Down'),
    #
    ax.bar(x + width / 2, final_data_dict["none_ac"], width,
           bottom=np.array(final_data_dict["left_ac"]) + np.array(final_data_dict["right_ac"]) + np.array(
               final_data_dict["up_ac"]) + np.array(final_data_dict["down_ac"]),
           color="#983600", edgecolor="black", label='Actual None'),
]
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.xticks(rotation=90)

plt.xlabel("Name of experiments")
plt.ylabel("Actual and predict direction (counts)")
plt.ylim(0, 1200)
# ax.bar_label(rect_1, padding=3, rotation=90, fontsize=8)
# ax.bar_label(rect_2, padding=3, rotation=90, fontsize=8)
# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

legend = plt.legend(loc='upper right', edgecolor="black", ncol=2, mode="expand")
legend.get_frame().set_alpha(0.1)
plt.rcParams["font.family"] = "Lato"
fig.tight_layout()

plt.savefig('comparison_between_detect_actual_direction.png', dpi=400)

plt.show()
