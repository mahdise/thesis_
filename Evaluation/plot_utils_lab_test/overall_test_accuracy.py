import numpy as np

from Evaluation.utils import read_data

data_of_position = read_data(
    "/Evaluation/StatisticalData/latest_all_experiment_position.csv")

data_of_direction = read_data(
    "/Evaluation/StatisticalData/latest_experiment_direction.csv")

data_of_angle = read_data(
    "/Evaluation/StatisticalData/latest_experiment_rotate_angle_latest.csv")

data_of_box_detect = read_data(
    "/Evaluation/StatisticalData/latest_experiment_box_detect_top_cam.csv")

# for arrow position
data_of_position = data_of_position.fillna(0)
data_accuracy_from_position = data_of_position["accuracy"].tolist()
__get_overall_accuracy_pos = sum(data_accuracy_from_position) / len(data_accuracy_from_position) * 100

# for arrow direction
data_of_direction = data_of_direction.fillna(0)
data_accuracy_from_direction = data_of_direction["accuracy"].tolist()
__get_overall_accuracy_direction = sum(data_accuracy_from_direction) / len(data_accuracy_from_direction) * 100

# for arrow detcetion pipeline
__get_accuracy_of_arrow_detection_pipeline = (__get_overall_accuracy_pos + __get_overall_accuracy_direction) / 2
print(__get_accuracy_of_arrow_detection_pipeline)

# for angle
data_error_angle = data_of_angle["mape"].tolist()
__get_overall_accuracy_angle = 100 - (sum(data_error_angle) / len(data_error_angle))

print(__get_overall_accuracy_angle)

# for detect box
data_error_box_detcte = data_of_box_detect["mape"].tolist()
__get_overall_accuracy_box_detction = 100 - (sum(data_error_box_detcte) / len(data_error_box_detcte))

print(__get_overall_accuracy_box_detction)

# overall orientation detction
final_accuracy = (
                         __get_accuracy_of_arrow_detection_pipeline + __get_overall_accuracy_angle + __get_overall_accuracy_box_detction) / 3
print(final_accuracy)

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

fig, ax = plt.subplots()
size = 0.3
vals = np.array([[90., 1.], [10., 0.]])
cmap = plt.get_cmap("tab20c")
outer_colors = ["#1E5162", "#FF8C00"]
inner_colors = ["#296E85", "#FF8C00"]
inner_colors_acngle = ["#338BA8", "#FF8C00"]
inner_colors_arrow = ["#ADD8E6", "#FF8C00"]

ax.pie([final_accuracy, 100 - final_accuracy], radius=1.5, colors=outer_colors,
       wedgeprops=dict(width=size, edgecolor='w'), labels=[round(final_accuracy), round(100 - final_accuracy)],
       labeldistance=0.85)
ax.pie([__get_overall_accuracy_box_detction, 100 - __get_overall_accuracy_box_detction], radius=1.5 - size,
       colors=inner_colors,
       wedgeprops=dict(width=size, edgecolor='w'),
       labels=[round(__get_overall_accuracy_box_detction), round(100 - __get_overall_accuracy_box_detction)],
       labeldistance=0.8)
ax.pie([__get_overall_accuracy_angle, 100 - __get_overall_accuracy_angle], radius=1.2 - size,
       colors=inner_colors_acngle,
       wedgeprops=dict(width=size, edgecolor='w'),
       labels=[round(__get_overall_accuracy_angle), round(100 - __get_overall_accuracy_angle)],
       labeldistance=0.8
       )
ax.pie([__get_accuracy_of_arrow_detection_pipeline, 100 - __get_accuracy_of_arrow_detection_pipeline],
       radius=0.89 - size, colors=inner_colors_arrow,
       wedgeprops=dict(width=size, edgecolor='w'),
       labels=[round(__get_accuracy_of_arrow_detection_pipeline),
               round(100 - __get_accuracy_of_arrow_detection_pipeline)],
       labeldistance=0.8
       )
ax.annotate('All values n percentage',
            xy=(1, 0),
            xytext=(-20, 20),
            horizontalalignment='right',
            verticalalignment='bottom')
plt.rcParams["font.family"] = "Lato"
plt.savefig('accuracy_overall.png', dpi=400)

plt.show()
