from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Evaluation.utils import read_data, label_transform_direction_back

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

column_name = "arrow_direction"
actual_value_column_name = "arrow_direction_gt_1"

list_of_value = list()
list_of_experiment_name = list()
list_actual_value = list()
std_pred_value_list = list()

experiment_name = ["Only one box", "Two boxes", "Three boxes",
                   "Environment clean", " Environment medium", " Environment Hard",
                   " Box own colour", "Box different colour ", "Distance <50cm",
                   " Distance >50cm", " Small size box", "Medium size box",
                   " Large size box", "Bright ambient lighting", "No ambient lighting"]

list_good_percentage = list()
list_mid_percentage = list()
list_bad_percentage = list()
for index, data in enumerate(all_data_frame):
    # data = data.fillna(method='ffill')
    # data = data.fillna(method='bfill')
    data = label_transform_direction_back(data, actual_value_column_name)

    list_data_pred = list(data[column_name])
    list_data_actual = list(data[actual_value_column_name + "_encode"])
    if list_data_actual[0] is None:
        percentage_of_good = 0
        percentage_of_bad = 0
    else:

        list_true_false = [True] * len(list_data_actual)

        for index, x in enumerate(list_data_actual):
            if list_data_pred[index] == x:
                list_true_false[index] = True
            else:
                list_true_false[index] = False
        unqiue_items_number = Counter(list_true_false)

        percentage_of_good = (unqiue_items_number[True] / len(list_data_actual)) * 100
        percentage_of_bad = (unqiue_items_number[False] / len(list_data_actual)) * 100

    list_good_percentage.append(percentage_of_good)
    list_bad_percentage.append(percentage_of_bad)

# prepare data
final_data_dict = {
    "experiment_name": experiment_name,
    "good": list_good_percentage,
    "bad": list_bad_percentage,

}
result_to_data_frame = pd.DataFrame(data=final_data_dict)
# print(result_to_data_frame)
csv_name = "dir_percenatage_of_good_bad_for_every_experiments.csv"
result_to_data_frame.to_csv(
    r'/home/mahdiislam/Mahdi/Biba/iris-parcel-orientation-detection/Evalouation/plots_custom_data/' + csv_name,
    index=True, header=True)
#

# plot functions


labels = result_to_data_frame["experiment_name"]
good_ = result_to_data_frame["good"]
bad_ = result_to_data_frame["bad"]

x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, good_, width, label='Good', color="#008080")
rects3 = ax.bar(x + width / 2, bad_, width, label='Bad', color="#ff8c00")

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.xticks(rotation=90)

plt.xlabel("Name of experiments")
plt.ylabel("Percentage values")

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
fig.tight_layout()

legend = plt.legend(loc='upper left', edgecolor="black")
legend.get_frame().set_alpha(0.1)
plt.rcParams["font.family"] = "Lato"
# plt.savefig('How_much_percentage_good_bad_direction.png', dpi=400)

plt.show()
