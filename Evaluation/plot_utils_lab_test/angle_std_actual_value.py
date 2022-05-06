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

actual_value_list = list()
pred_value_list = list()
std_actual_value_list = list()
std_pred_value_list = list()
for data in all_data_frame:
    data = data.fillna(method='ffill')
    list_data_pred = list(data[column_name])
    list_data_actual = list(data[actual_value_column_name])

    pred_mean = np.mean(list_data_pred)
    actual_mean = np.mean(list_data_actual)

    pred_value_list.append(pred_mean)
    actual_value_list.append(actual_mean)

    pred_std = np.std(list_data_pred)
    actual_std = np.std(list_data_actual)

    std_pred_value_list.append(pred_std)
    std_actual_value_list.append(actual_std)

# prepare data
final_data_dict = {
    "Detect": pred_value_list,
    "Actual": actual_value_list,
    "pred_std": std_pred_value_list,
    "actual_std": std_actual_value_list,

}

experiment_name = ["Only one box", "Two boxes", "Three boxes",
                   "Environment clean", " Environment medium", " Environment Hard",
                   " Box own colour", "Box different colour ", "Distance <50cm",
                   " Distance >50cm", " Small size box", "Medium size box",
                   " Large size box", "Bright ambient lighting", "No ambient lighting"]

df_ = pd.DataFrame(data=final_data_dict, index=experiment_name)
df_[['Detect', 'Actual']].plot(kind='bar',
                               yerr=df_[['pred_std', 'actual_std']].values.T,
                               alpha=0.8,
                               error_kw=dict(ecolor='k'),
                               color=['#008080', '#32CD32'])
plt.xlabel("Name of experiments")
plt.ylabel("STD and actual value on angle "
           "\n"
           "(degree)")
# plt.title("Relation between various experiments and std with actual angle")
plt.xticks(rotation=90)
plt.tight_layout()

legend = plt.legend(loc='upper left', edgecolor="black")
legend.get_frame().set_alpha(0.1)
plt.rcParams["font.family"] = "Lato"
plt.savefig('bar_plot_with_error_bars.png', dpi=400)

# legend.get_frame().set_facecolor((0, 0, 1, 0.1))
plt.show()

