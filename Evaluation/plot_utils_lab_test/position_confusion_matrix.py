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

column_name = "arrow_position"
actual_value_column_name = "arrow_position_gt_1"

list_of_value = list()
list_of_experiment_name = list()
list_actual_value = list()
std_pred_value_list = list()

experiment_name = ["Only one box", "Two boxes", "Three boxes",
                   "Environment clean", " Environment medium", " Environment Hard",
                   " Box own colour", "Box different colour ", "Distance <50cm",
                   " Distance >50cm", " Small size box", "Medium size box",
                   " Large size box", "Bright ambient lighting", "No ambient lighting"]

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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label('Arrow position counts')

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        print("..............")
        print(i)
        print(j)
        print(cm[i, j])

        print("____")
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True arrow position')
    plt.xlabel('Predicted arrow position')
    plt.savefig('confusion_matrix.png', dpi=400)


# Compute confusion matrix
cnf_matrix = confusion_matrix(list_actual_value_all, list_pridict_value_all)
cnf_matrix_ = confusion_matrix(list_actual_value_all, list_pridict_value_all).ravel()

FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)
list_demo_none = [TP[0], FP[0], TN[0], FN[0]]
list_demo_top = [TP[1], FP[1], TN[1], FN[1]]
list_demo_side = [TP[2], FP[2], TN[2], FN[2]]

# Precision = TP/TP+FP
Precision_top = list_demo_top[0] / (list_demo_top[0] + list_demo_top[1])
Precision_side = list_demo_side[0] / (list_demo_side[0] + list_demo_side[1])
Precision_none = list_demo_none[0] / (list_demo_none[0] + list_demo_none[1])

# Recall = TP/TP+FN

Recall_top = list_demo_top[0] / (list_demo_top[0] + list_demo_top[3])
Recall_side = list_demo_side[0] / (list_demo_side[0] + list_demo_side[3])
Recall_none = list_demo_none[0] / (list_demo_none[0] + list_demo_none[3])

np.set_printoptions(precision=2)
class_names = ["None", "Top", "Side"]
# Plot non-normalized confusion matrix
plt.figure()
# plt.figure(figsize=(8,7))

plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

plt.show()
