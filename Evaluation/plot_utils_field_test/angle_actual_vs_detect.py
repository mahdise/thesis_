import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Evaluation.utils import read_data

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

column_name = "rotate_angle"
actual_value_column_name = "angle_gt_1"

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
    data = data.fillna(method='ffill')
    data = data.fillna(method='bfill')

    list_data_pred = list(data[column_name])
    list_data_actual = list(data[actual_value_column_name])

    mean_value_pred = statistics.mean(list_data_pred)
    mean_value_actual = statistics.mean(list_data_actual)

    list_good_percentage.append(round(mean_value_pred))
    list_bad_percentage.append(round(mean_value_actual))

# prepare data
final_data_dict = {
    "experiment_name": data_frame_experiment_label,
    "good": list_good_percentage,
    "bad": list_bad_percentage,

}
result_to_data_frame = pd.DataFrame(data=final_data_dict)
print(result_to_data_frame)
# csv_name = "angle_avg_actual_value.csv"
# result_to_data_frame.to_csv(
#     r'/home/mahdiislam/Mahdi/Biba/iris-parcel-orientation-detection/Evalouation/plots_custom_data/' + csv_name,
#     index=True, header=True)


# plot functions


labels = result_to_data_frame["experiment_name"]
good_ = result_to_data_frame["good"]
bad_ = result_to_data_frame["bad"]

x = np.arange(len(labels))  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, good_, width, label='Detect', color="#008080")
rects3 = ax.bar(x + width / 2, bad_, width, label='Actual', color="#32CD32")

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.xticks(rotation=90)

plt.xlabel("Name of experiments")
plt.ylabel("Average values")
plt.ylim(0, 200)
ax.bar_label(rects1, padding=3, rotation=90, fontsize=8)
ax.bar_label(rects3, padding=3, rotation=90, fontsize=8)
fig.tight_layout()

legend = plt.legend(loc='upper right', edgecolor="black")
legend.get_frame().set_alpha(0.1)
plt.rcParams["font.family"] = "Lato"
plt.savefig('actual_vs_avg_angle.png', dpi=400)

plt.show()
