# data read of statistical

import matplotlib.pyplot as plt
import numpy as np

from Evaluation.utils import read_data

data_of_st = read_data(
    "/Evaluation/statistical_data_field_test/latest_experiment_arroe_dir.csv")

final_data_dict = {
    "experiment_name": data_of_st["name_of_experiment"],
    "accuracy": data_of_st["accuracy"],
    "recall": data_of_st["reacall"],
    "precision": data_of_st["priccision"],

}

x = np.arange(len(final_data_dict["experiment_name"]))  # the label locations
width = 0.25  # the width of the bars
x_axis = range(0, 15)

fig, ax = plt.subplots()
rects1 = ax.plot(x, final_data_dict["accuracy"], '-ok', label='Accuracy', color="#orange"
                                                                                "")
# rects2 = ax.plot(x, final_data_dict["recall"], '-p',label='Recall', color="#32CD32")
# rects3 = ax.plot(x, final_data_dict["precision"], label='Precision', color="#ff8c00")

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(final_data_dict["experiment_name"])
ax.legend()
plt.xticks(rotation=90)

plt.xlabel("Name of experiments")
plt.ylabel("Values")
plt.ylim(-0.1, 1.5)
# ax.bar_label(rects1, padding=3, rotation=90, fontsize=8)
# ax.bar_label(rects3, padding=3, rotation=90, fontsize=8)
fig.tight_layout()

legend = plt.legend(loc='upper right', edgecolor="black")
legend.get_frame().set_alpha(0.1)
plt.rcParams["font.family"] = "Lato"
plt.savefig('accuracy_direction.png', dpi=400)

plt.show()
