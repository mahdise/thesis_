from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Evaluation.utils import read_data, label_transform_position_back

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

column_name = "arrow_position"
actual_value_column_name = "arrow_position_gt_1"

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
# import pandas as pd
#
# df = pd.DataFrame(dict(
#     A=[1, 2, 3, 4],
#     B=[2, 3, 4, 5],
#     C=[3, 4, 5, 6],
#     D=[4, 5, 6, 7]))
#
# import matplotlib.pyplot as plt
#
# fig = plt.figure(figsize=(20, 10))
#
# ab_bar_list = [plt.bar([0, 1, 2, 3], df.B, width=0.2, label="men"),
#                plt.bar([0, 1, 2, 3], df.A, width=0.2, label="vvvmen")]
#
# cd_bar_list = [plt.bar([0, 1, 2, 3], df.D, width=-0.2, label="ggggmen"),
#                plt.bar([0, 1, 2, 3], df.C, width=-0.2, label="vvvvmen")]
#
# plt.legend()
# plt.show()


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
    ax.bar(x + width / 2, final_data_dict["top_actual"], width,
           bottom=final_data_dict["side_actual"], color="#32CD32", edgecolor="black", label='Actua Top'),

    ax.bar(x + width / 2, final_data_dict["none_actual"], width,
           bottom=np.array(final_data_dict["side_actual"]) + np.array(final_data_dict["top_actual"]),
           color="#ff8c00", edgecolor="black", label='Actual None')

]
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.xticks(rotation=90)

plt.xlabel("Name of experiments")
plt.ylabel("Actual and predict position (counts)")
plt.ylim(0, 1200)
# ax.bar_label(rect_1, padding=3, rotation=90, fontsize=8)
# ax.bar_label(rect_2, padding=3, rotation=90, fontsize=8)
# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
fig.tight_layout()

legend = plt.legend(loc='upper right', edgecolor="black")
legend.get_frame().set_alpha(0.1)
plt.rcParams["font.family"] = "Lato"
plt.savefig('comparison_between_detect_actual_position.png', dpi=400)

plt.show()
