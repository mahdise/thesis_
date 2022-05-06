import matplotlib.pyplot as plt

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

for index, data in enumerate(all_data_frame):
    data = data.fillna(method='ffill')
    data = data.fillna(method='bfill')

    list_data_pred = list(data[column_name])
    x = range(0, len(list_data_pred))

    list_data_actual = list(data[actual_value_column_name])

    fig, ax = plt.subplots()
    rects1 = ax.plot(x, list_data_pred, label='Detect', color="#008080")
    rects3 = ax.plot(x, list_data_actual, label='Actual', color="#ff8c00")

    ax.legend()

    name_of_x_axis = experiment_name[index] + " test"

    plt.xlabel(name_of_x_axis)
    plt.ylabel("Angle (Degree)")

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)
    fig.tight_layout()

    legend = plt.legend(loc='upper left', edgecolor="black")
    legend.get_frame().set_alpha(0.1)
    plt.rcParams["font.family"] = "Lato"

    save_name = name_of_x_axis + ".png"
    plt.savefig(save_name)

    plt.show()
