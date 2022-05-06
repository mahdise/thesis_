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

experiment_name = ["Vertical", "Horizental ", "Z-Horizental ",
                   "Short distance ", " Medium distance",
                   "Long distance",
                   "Attach box", "Small size ",
                   "Medium size ", " Large size "]

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

# # Data
# aluminum = np.array([6.4e-5 , 3.01e-5 , 2.36e-5, 3.0e-5, 7.0e-5, 4.5e-5, 3.8e-5,
#                      4.2e-5, 2.62e-5, 3.6e-5])
# copper = np.array([4.5e-5 , 1.97e-5 , 1.6e-5, 1.97e-5, 4.0e-5, 2.4e-5, 1.9e-5,
#                    2.41e-5 , 1.85e-5, 3.3e-5 ])
# steel = np.array([3.3e-5 , 1.2e-5 , 0.9e-5, 1.2e-5, 1.3e-5, 1.6e-5, 1.4e-5,
#                   1.58e-5, 1.32e-5 , 2.1e-5])
#
# # Calculate the average
# aluminum_mean = np.mean(aluminum)
# copper_mean = np.mean(copper)
# steel_mean = np.mean(steel)
#
#
# # Calculate the standard deviation
# aluminum_std = np.std(aluminum)
# copper_std = np.std(copper)
# steel_std = np.std(steel)
#
#
# # Define labels, positions, bar heights and error bar heights
# labels = ['Aluminum', 'Copper', 'Steel']
# x_pos = np.arange(len(labels))
# CTEs = [aluminum_mean, copper_mean, steel_mean]
# error = [aluminum_std, copper_std, steel_std]
#
#
# # Build the plot
# fig, ax = plt.subplots()
# ax.bar(x_pos, CTEs,
#        yerr=error,
#        align='center',
#        alpha=0.5,
#        ecolor='k',
#        capsize=10,
#        color=['black', 'red', 'green'])
#
#
# ax.set_ylabel('Coefficient of Thermal Expansion ')
# ax.set_xlabel('Coefficient of Thermal Expansion ')
# ax.set_xticks(x_pos)
# ax.set_xticklabels(labels)
# ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
# ax.yaxis.grid(False)
#
# # Save the figure and show
# plt.tight_layout()
# # plt.savefig('bar_plot_with_error_bars.png')
# plt.show()
