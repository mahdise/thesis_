from sklearn import metrics

from Evaluation.statical_analysis.calclate_recall_precission import calculate_pre_recall_accuracy_fp
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
    import sklearn.metrics as metrics

    # true_class_list  =  1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 2, 3, 1, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 2, 2, 2, 2, 0, 0, 0, 3, 3, 3, 3, 3, 2]
    #
    # predicted_class_list = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 2, 3, 1, 0, 1, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 2, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 2, 2, 2, 2, 0, 0, 0, 3, 3, 3, 3, 3, 2]
    falsepr, truepr, threshold = metrics.roc_curve(list_data_actual, list_data_pred, pos_label=2)
    roc_auc = metrics.auc(falsepr, truepr)
    # method I: plt
    import matplotlib.pyplot as plt

    plt.title('Receiver Operating Characteristic')
    plt.plot(falsepr, truepr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    for x in list_data_pred:
        list_pridict_value_all.append(x)

    for y in list_data_actual:
        list_actual_value_all.append(y)

import sklearn.metrics as metrics

# true_class_list  =  1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 2, 3, 1, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 2, 2, 2, 2, 0, 0, 0, 3, 3, 3, 3, 3, 2]
#
# predicted_class_list = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 2, 3, 1, 0, 1, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 2, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 2, 2, 2, 2, 0, 0, 0, 3, 3, 3, 3, 3, 2]
falsepr, truepr, threshold = metrics.roc_curve(list_actual_value_all, list_pridict_value_all, pos_label=1)
roc_auc = metrics.auc(falsepr, truepr)
# method I: plt
import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')
plt.plot(falsepr, truepr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# fpr, tpr, _ = metrics.roc_curve(list_actual_value_all,  list_pridict_value_all)


precission, recall, accuracy, false_positive_rate, true_positive_rate, dict_new = calculate_pre_recall_accuracy_fp(
    list_actual_value_all, list_pridict_value_all)


def roc_auc_score_multiclass(actual_class, pred_class, average="macro"):
    # creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        # creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        # marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        # using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = metrics.roc_auc_score(new_actual_class, new_pred_class, average=average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict


n_classes = [0.0, 1.0, 2.0]

auc = roc_auc_score_multiclass(list_actual_value_all, list_pridict_value_all)
# false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(actual_value_column_name, predicted_column_value_name)


colors = ["red", "blue", "orange"]
asd = dict_new[1.0]["fp"]

plt.plot(0.6, 0.9, color="orange", lw=1.5,
         label='ROC curve of class {0} (area = {1:0.2f})'
               ''.format(0, 12))
# for index_class,value   in enumerate(n_classes):
#     plt.plot(dict_new[value]["fp"], dict_new[value]["tp"], color=colors[index_class], lw=1.5,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#                    ''.format(value, auc[value]))
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
# plt.xlim([-0.05, 1.0])
# plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()
