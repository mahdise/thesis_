import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Evaluation.utils import read_data

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

column_name = "rotate_angle"
actual_value_column_name = "angle_gt_1"

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
    data = data.fillna(method='ffill')
    data = data.fillna(method='bfill')

    list_data_pred = list(data[column_name])
    list_data_actual = list(data[actual_value_column_name])

    diff_between_actual_pred = list(data[actual_value_column_name].sub(data[column_name], axis=0))
    gdk = [(x / 100) * list_data_pred[0] for x in diff_between_actual_pred]
    abs_diff_between_actual_pred = [abs(ele) for ele in diff_between_actual_pred]

    data["abs_diff_between_actual_pred_angle"] = abs_diff_between_actual_pred
    data["abs_diff_between_actual_pred_angle"] = data["abs_diff_between_actual_pred_angle"]

    min_value = min(data["abs_diff_between_actual_pred_angle"])
    max_value = max(data["abs_diff_between_actual_pred_angle"])
    avg_value = data["abs_diff_between_actual_pred_angle"].mean()
    avg_value_75 = np.percentile(list(data["abs_diff_between_actual_pred_angle"]), 75)
    mid_val = avg_value - min_value

    # if defference is 5 or less good
    # if diff is gt 6 or less then 20 mid
    # if diff is gt 20

    good_percentage = data["abs_diff_between_actual_pred_angle"] <= 2
    good_percentage = good_percentage.mean() * 100

    medium_percentage = (data["abs_diff_between_actual_pred_angle"] >= 3) & (
            data["abs_diff_between_actual_pred_angle"] <= 10)
    medium_percentage = medium_percentage.mean() * 100

    bad_percentage = data["abs_diff_between_actual_pred_angle"] >= 10
    bad_percentage = bad_percentage.mean() * 100

    list_good_percentage.append(good_percentage)
    list_mid_percentage.append(medium_percentage)
    list_bad_percentage.append(bad_percentage)

# prepare data
final_data_dict = {
    "experiment_name": experiment_name,
    "good": list_good_percentage,
    "bad": list_bad_percentage,
    "mid": list_mid_percentage

}
result_to_data_frame = pd.DataFrame(data=final_data_dict)

labels = result_to_data_frame["experiment_name"]
good_ = result_to_data_frame["good"]
mid_ = result_to_data_frame["mid"]
bad_ = result_to_data_frame["bad"]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, good_, width, label='Good (<2)', color="#008080")
rects2 = ax.bar(x, mid_, width, label='Medium (3-10)', color="#32CD32")
rects3 = ax.bar(x + width, bad_, width, label='Bad (10<)', color="#ff8c00")

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

legend = plt.legend(loc='upper right', edgecolor="black")
legend.get_frame().set_alpha(0.1)
plt.rcParams["font.family"] = "Lato"
plt.savefig('How_much_percentage_good_bad_angle.png', dpi=400)

plt.show()
