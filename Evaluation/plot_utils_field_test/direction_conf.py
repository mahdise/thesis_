import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from Evaluation.utils import read_data

"""
https://stackoverflow.com/questions/56815698/valueerror-only-one-class-present-in-y-true-roc-auc-score-is-not-defined-in-th/56822934
You cannot have an ROC curve without both positive and negative examples in your dataset. With only one class in the dataset, you cannot measure your false-positive rate, and therefore cannot plot an ROC curve. This is why you get this error message.
"""
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

column_name = "arrow_direction"
actual_value_column_name = "arrow_direction_gt_1"

list_of_value = list()
list_of_experiment_name = list()
list_actual_value = list()
std_pred_value_list = list()

experiment_name = ["Vertical", "Horizental ", "Z-Horizental ",
                   "Short distance ", " Medium distance",
                   "Long distance",
                   "Attach box", "Small size ",
                   "Medium size ", " Large size "]

list_actual_value_all = list()
list_pridict_value_all = list()

for index, data in enumerate(all_data_frame):
    data = data.fillna(method='ffill')
    data = data.fillna(method='bfill')
    # data = label_transform_position_back(data, actual_value_column_name)

    list_data_pred = list(data[column_name + "_encode"])
    list_data_actual = list(data[actual_value_column_name])

    for x in list_data_pred:
        list_pridict_value_all.append(x)

    for y in list_data_actual:
        list_actual_value_all.append(y)

for sd_, x in enumerate(list_actual_value_all):
    if x == "left":
        list_actual_value_all[sd_] = 1.0
    else:
        list_actual_value_all[sd_] = float(x)

for df, y in enumerate(list_pridict_value_all):
    if y == "left":
        list_pridict_value_all[df] = 1.0

    else:
        list_pridict_value_all[df] = float(y)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label('Arrow direction counts')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True arrow direction')
    plt.xlabel('Predicted arrow direction')
    plt.savefig('confusion_matrix_direction.png', dpi=400)


# Compute confusion matrix
cnf_matrix = confusion_matrix(list_actual_value_all, list_pridict_value_all)
np.set_printoptions(precision=2)
class_names = ["None", "Left", "Right", "Up", "Down"]
# Plot non-normalized confusion matrix
plt.figure()
# plt.figure(figsize=(8,7))

plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

plt.show()
