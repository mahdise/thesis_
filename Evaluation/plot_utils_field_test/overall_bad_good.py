from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Evaluation.utils import read_data, label_transform_position_back, label_transform_direction_back

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
column_name_pos = "arrow_position"
actual_value_column_name_pos = "arrow_position_gt_1"
column_name_dir = "arrow_direction"
actual_value_column_name_dir = "arrow_direction_gt_1"
column_name_angle = "rotate_angle"
actual_value_column_name_angle = "angle_gt_1"

list_of_value = list()
list_of_experiment_name = list()
list_actual_value = list()
std_pred_value_list = list()

data_frame_experiment_label = ["Vertical", "Horizental ", "Z-Horizental ",
                               "Short distance ", " Medium distance",
                               "Long distance",
                               "Attach box", "Small size ",
                               "Medium size ", " Large size "]

list_good_percentage = list()
list_mid_percentage = list()
list_bad_percentage = list()
for index, data in enumerate(all_data_frame):
    # data = data.fillna(method='ffill')
    # data = data.fillna(method='bfill')
    data_pos = label_transform_position_back(data, actual_value_column_name_pos)
    data_dir = label_transform_direction_back(data, actual_value_column_name_dir)

    list_data_pred_pos = list(data_pos[column_name_pos])
    list_data_actual_pos = list(data_pos[actual_value_column_name_pos + "_encode"])

    list_data_pred_dir = list(data_dir[column_name_dir])
    list_data_actual_dir = list(data_dir[actual_value_column_name_dir + "_encode"])

    list_data_pred_angle = list(data[column_name_angle])
    list_data_actual_angle = list(data[actual_value_column_name_angle])

    diff_between_actual_pred = list(data[actual_value_column_name_angle].sub(data[column_name_angle], axis=0))
    gdk = [(x / 100) * list_data_pred_angle[0] for x in diff_between_actual_pred]
    abs_diff_between_actual_pred = [abs(ele) for ele in diff_between_actual_pred]
    data["abs_diff_between_actual_pred_angle"] = abs_diff_between_actual_pred
    data["abs_diff_between_actual_pred_angle"] = data["abs_diff_between_actual_pred_angle"]

    if list_data_actual_pos[0] is None or list_data_actual_dir[0] is None:
        # percentage_of_good = 0
        # percentage_of_bad = 0
        # percentage_of_mid = 0

        good_percentage = data["abs_diff_between_actual_pred_angle"] <= 2
        percentage_of_good = good_percentage.mean() * 100

        medium_percentage = (data["abs_diff_between_actual_pred_angle"] >= 3) & (
                data["abs_diff_between_actual_pred_angle"] <= 10)
        percentage_of_mid = medium_percentage.mean() * 100

        bad_percentage = data["abs_diff_between_actual_pred_angle"] >= 10
        percentage_of_bad = bad_percentage.mean() * 100

    else:

        list_true_false_pos = [True] * len(list_data_actual_pos)
        list_true_false_dir = [True] * len(list_data_actual_dir)
        list_true_false_angle = [True] * len(list_data_actual_dir)

        for index, x in enumerate(list_data_actual_pos):
            if list_data_pred_pos[index] == x:
                list_true_false_pos[index] = True
            else:
                list_true_false_pos[index] = False

        for index, x in enumerate(list_data_actual_dir):
            if list_data_pred_dir[index] == x:
                list_true_false_dir[index] = True
            else:
                list_true_false_dir[index] = False

        for index, x in enumerate(list_data_actual_dir):
            if data["abs_diff_between_actual_pred_angle"][index] <= 5:
                list_true_false_angle[index] = True
            else:
                list_true_false_angle[index] = False

        new_combine_true_false = ["True"] * len(list_data_actual_pos)

        for index, y in enumerate(new_combine_true_false):
            if list_true_false_pos[index] == True and list_true_false_dir[index] == True and \
                    data["abs_diff_between_actual_pred_angle"][index] <= 5:
                new_combine_true_false[index] = "True"
            elif list_true_false_pos[index] == True or list_true_false_dir[index] == True:
                new_combine_true_false[index] = "Mid"


            else:
                new_combine_true_false[index] = "False"

        unqiue_items_number = Counter(new_combine_true_false)

        percentage_of_good = (unqiue_items_number["True"] / len(new_combine_true_false)) * 100
        percentage_of_mid = (unqiue_items_number["Mid"] / len(new_combine_true_false)) * 100
        percentage_of_bad = (unqiue_items_number["False"] / len(new_combine_true_false)) * 100

    list_good_percentage.append(percentage_of_good)
    list_mid_percentage.append(percentage_of_mid)
    list_bad_percentage.append(percentage_of_bad)

# prepare data
final_data_dict = {
    "experiment_name": data_frame_experiment_label,
    "good": list_good_percentage,
    "bad": list_bad_percentage,
    "mid": list_mid_percentage

}
result_to_data_frame = pd.DataFrame(data=final_data_dict)
print(result_to_data_frame)
# csv_name = "overall_percenatage_of_good_bad_for_every_experiments.csv"
# result_to_data_frame.to_csv(
#     r'/home/mahdiislam/Mahdi/Biba/iris-parcel-orientation-detection/Evalouation/plots_custom_data/' + csv_name,
#     index=True, header=True)


# plot functions


labels = result_to_data_frame["experiment_name"]
good_ = result_to_data_frame["good"]
mid_ = result_to_data_frame["mid"]
bad_ = result_to_data_frame["bad"]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, good_, width, label='Good', color="#008080")
rects2 = ax.bar(x, mid_, width, label='Medium', color="#32CD32")
rects3 = ax.bar(x + width, bad_, width, label='Bad', color="#ff8c00")

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.xticks(rotation=90)

plt.xlabel("Name of experiments")
plt.ylabel("Overall orientation"
           "\n"
           "(percentage)")

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
fig.tight_layout()

legend = plt.legend(loc='upper left', edgecolor="black")
legend.get_frame().set_alpha(0.1)
plt.rcParams["font.family"] = "Lato"
plt.savefig('How_much_percentage_good_bad_overall.png', dpi=400)

plt.show()
