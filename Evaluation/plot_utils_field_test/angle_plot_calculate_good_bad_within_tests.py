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

    good_percentage = data["abs_diff_between_actual_pred_angle"] < 2
    good_percentage = good_percentage.mean() * 100

    medium_percentage = (data["abs_diff_between_actual_pred_angle"] >= 2) & (
            data["abs_diff_between_actual_pred_angle"] <= 10)
    medium_percentage = medium_percentage.mean() * 100

    bad_percentage = data["abs_diff_between_actual_pred_angle"] >= 10
    bad_percentage = bad_percentage.mean() * 100

    list_good_percentage.append(good_percentage)
    list_mid_percentage.append(medium_percentage)
    list_bad_percentage.append(bad_percentage)

# prepare data
final_data_dict = {
    "experiment_name": data_frame_experiment_label,
    "good": list_good_percentage,
    "bad": list_bad_percentage,
    "mid": list_mid_percentage

}
result_to_data_frame = pd.DataFrame(data=final_data_dict)
print(result_to_data_frame)
# print(result_to_data_frame)
# csv_name = "angle_percenatage_of_good_bad_for_every_experiments.csv"
# result_to_data_frame.to_csv(
#     r'/home/mahdiislam/Mahdi/Biba/iris-parcel-orientation-detection/Evalouation/plot_data/' + csv_name,
#     index=True, header=True)


# plot functions


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
