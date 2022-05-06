from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Evaluation.utils import read_data, label_transform_position_back

data_frame_1111 = read_data(
    "/Evaluation/ProcessedData/preprocessed_gt_orientation_result_1111.csv")
data_frame_1122 = read_data(
    "/Evaluation/ProcessedData/preprocessed_gt_orientation_result_1122.csv")
data_frame_1133 = read_data(
    "/Evaluation/ProcessedData/preprocessed_gt_orientation_result_1133.csv")
data_frame_2210 = read_data(
    "/Evaluation/ProcessedData/preprocessed_gt_orientation_result_2210.csv")
data_frame_2211 = read_data(
    "/Evaluation/ProcessedData/preprocessed_gt_orientation_result_2211.csv")
data_frame_2212 = read_data(
    "/Evaluation/ProcessedData/preprocessed_gt_orientation_result_2212.csv")
data_frame_3311 = read_data(
    "/Evaluation/ProcessedData/preprocessed_gt_orientation_result_3311.csv")
data_frame_3312 = read_data(
    "/Evaluation/ProcessedData/preprocessed_gt_orientation_result_3312.csv")
data_frame_4411 = read_data(
    "/Evaluation/ProcessedData/preprocessed_gt_orientation_result_4411.csv")
data_frame_4412 = read_data(
    "/Evaluation/ProcessedData/preprocessed_gt_orientation_result_4412.csv")
data_frame_5511 = read_data(
    "/Evaluation/ProcessedData/preprocessed_gt_orientation_result_5511.csv")
data_frame_5512 = read_data(
    "/Evaluation/ProcessedData/preprocessed_gt_orientation_result_5512.csv")
data_frame_5513 = read_data(
    "/Evaluation/ProcessedData/preprocessed_gt_orientation_result_5513.csv")
data_frame_6611 = read_data(
    "/Evaluation/ProcessedData/preprocessed_gt_orientation_result_6611.csv")
data_frame_6612 = read_data(
    "/Evaluation/ProcessedData/preprocessed_gt_orientation_result_6612.csv")

all_data_frame = [data_frame_1111, data_frame_1122, data_frame_1133,
                  data_frame_2210, data_frame_2211, data_frame_2211,
                  data_frame_3311, data_frame_3312, data_frame_4411,
                  data_frame_4412, data_frame_5511, data_frame_5512,
                  data_frame_5513, data_frame_6611, data_frame_6612, ]

column_name = "arrow_position"
actual_value_column_name = "arrow_position_gt_1"

list_of_value = list()
list_of_experiment_name = list()
list_actual_value = list()
std_pred_value_list = list()

experiment_name = ["Only one box", "Two boxes", "Three boxes",
                   "Environment clean", " Environment medium", " Environment Hard",
                   " Box own colour", "Box different colour ", "Distance <50cm",
                   " Distance >50cm", " Small size box", "Medium size box",
                   " Large size box", "Bright ambient lighting", "No ambient lighting"]

list_None_count_pred = list()
list_side_count_pred = list()
list_top_count_pred = list()

list_None_count_actual = list()
list_side_count_actual = list()
list_top_count_actual = list()
for index, data in enumerate(all_data_frame):
    # data = data.fillna(method='ffill')
    # data = data.fillna(method='bfill')
    data = label_transform_position_back(data, actual_value_column_name)

    list_data_pred = list(data[column_name])
    list_data_actual = list(data[actual_value_column_name + "_encode"])

    unqiue_items_number_pred = Counter(list_data_pred)
    unqiue_items_number_actual = Counter(list_data_actual)

    side_value_pred = unqiue_items_number_pred.get("Side") or 0
    top_value_pred = unqiue_items_number_pred.get("Top") or 0
    nan_value_pred = unqiue_items_number_pred.get(np.nan) or 0

    list_None_count_pred.append(nan_value_pred)
    list_side_count_pred.append(side_value_pred)
    list_top_count_pred.append(top_value_pred)

    side_value_actual = unqiue_items_number_actual.get("Side") or 0
    top_value_actual = unqiue_items_number_actual.get("Top") or 0
    nan_value_actual = unqiue_items_number_actual.get(np.nan) or 0

    list_None_count_actual.append(nan_value_actual)
    list_side_count_actual.append(side_value_actual)
    list_top_count_actual.append(top_value_actual)

# prepare data
final_data_dict = {
    "experiment_name": experiment_name,
    "side_pred": list_side_count_pred,
    "top_pred": list_top_count_pred,
    "none_pred": list_None_count_pred,
    "side_actual": list_side_count_actual,
    "top_actual": list_top_count_actual,
    "none_actual": list_None_count_actual,

}
result_to_data_frame = pd.DataFrame(data=final_data_dict)
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
    ax.bar(x - width / 2, final_data_dict["side_pred"], width, label='Predict Side', color="#008080"),
    ax.bar(x - width / 2, final_data_dict["top_pred"], width, bottom=final_data_dict["side_pred"],
           label='Predict Top', color="#32CD32"),

    ax.bar(x - width / 2, final_data_dict["none_pred"], width,
           bottom=np.array(final_data_dict["side_pred"]) + np.array(final_data_dict["top_pred"]),
           label='Predict None', color="#ff8c00"),
]

rect_2 = [
    ax.bar(x + width / 2, final_data_dict["side_actual"], width, color="#008080", edgecolor="black", label='Actuual '
                                                                                                           'Side'),
    ax.bar(x + width / 2, final_data_dict["top_actual"], width, bottom=final_data_dict["side_actual"], color="#32CD32",
           edgecolor="black",
           label='Actua Top'
           ),

    ax.bar(x + width / 2, final_data_dict["none_actual"], width,
           bottom=np.array(final_data_dict["side_actual"]) + np.array(final_data_dict["top_actual"]), color="#ff8c00",
           edgecolor="black",
           label='Actual None'
           )

]
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.xticks(rotation=90)

plt.xlabel("Name of experiments")
plt.ylabel("Actual and predict position (counts)")
plt.ylim(0, 1100)
asd = range(0, 15)

fig.tight_layout()

legend = plt.legend(loc='upper right', edgecolor="black")
legend.get_frame().set_alpha(0.1)
plt.rcParams["font.family"] = "Lato"
# plt.savefig('comparison_between_detect_actual_position.png', dpi=400)

plt.show()
