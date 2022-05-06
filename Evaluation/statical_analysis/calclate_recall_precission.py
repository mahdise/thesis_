import math
from sklearn.metrics import mean_squared_error
import math

from sklearn.metrics import mean_squared_error


def perf_measure(y_actual, y_pred):
    class_id = set(y_actual)
    tp = []
    fp = []
    tn = []
    fn = []

    for index, _id in enumerate(class_id):
        tp.append(0)
        fp.append(0)
        tn.append(0)
        fn.append(0)
        for i in range(len(y_pred)):
            a = y_actual[i]
            b = y_pred[i]
            if y_actual[i] == y_pred[i] == _id:
                tp[index] += 1
            if y_pred[i] == _id and y_actual[i] != y_pred[i]:
                fp[index] += 1
            if y_actual[i] == y_pred[i] != _id:
                tn[index] += 1
            if y_pred[i] != _id and y_actual[i] != y_pred[i]:
                fn[index] += 1

    return class_id, tp, fp, tn, fn


import numpy as np


def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100


def calculate_mse_rmse_r2_score(actual_value, predicted_value):
    mse = mean_squared_error(actual_value, predicted_value)
    rmse = math.sqrt(mean_squared_error(actual_value, predicted_value))

    mape_ = mape(actual_value, predicted_value)

    return mse, rmse, mape_


def calculate_pre_recall_accuracy(actual_value_, predicted_value_):
    actual_value_ = [float(i) for i in actual_value_]
    predicted_value_ = [float(i) for i in predicted_value_]

    id_, tp, fp, tn, fn = perf_measure(actual_value_, predicted_value_)
    try:
        precission = (tp[0] / (tp[0] + fp[0]))
    except:
        precission = None
    try:

        recall = (tp[0] / (tp[0] + fn[0]))
    except:
        recall = None

    try:
        ## calculate accuracy
        accuracy = len(
            [actual_value_[i] for i in range(0, len(actual_value_)) if actual_value_[i] == predicted_value_[i]]) / len(
            actual_value_)
    except:
        accuracy = None

    return precission, recall, accuracy


def calculate_pre_recall_accuracy_fp(actual_value_, predicted_value_):
    actual_value_ = [float(i) for i in actual_value_]
    predicted_value_ = [float(i) for i in predicted_value_]

    id_, tp, fp, tn, fn = perf_measure(actual_value_, predicted_value_)
    dict_new = dict()
    for index_, id_sep in enumerate(id_):
        #https://www.pico.net/kb/what-is-a-false-positive-rate/
        fpr_ = fp[index_]/(fp[index_]+tn[index_])
        tpr_ = tp[index_]/(tp[index_]+fn[index_])
        dict_new[id_sep] = {
            "tp": tpr_,
            "fp": fpr_,
            "tn": tn[index_],
            "fn": fn[index_],
        }

    try:
        precission = (tp[0] / (tp[0] + fp[0]))
    except:
        precission = None
    try:

        recall = (tp[0] / (tp[0] + fn[0]))
    except:
        recall = None

    try:
        ## calculate accuracy
        accuracy = len(
            [actual_value_[i] for i in range(0, len(actual_value_)) if actual_value_[i] == predicted_value_[i]]) / len(
            actual_value_)
    except:
        accuracy = None

    return precission, recall, accuracy, fp, tp, dict_new


if __name__ == '__main__':
    predicted_column_value_name = [1, 2, 1, 7]
    actual_value_column_name = [1, 2, 1, 1]
    # mse = mean_squared_error(actual_value_column_name, predicted_column_value_name)
    # rmse = math.sqrt(mean_squared_error(actual_value_column_name, predicted_column_value_name))
    # r2score = round(r2_score(actual_value_column_name, predicted_column_value_name), 2)
    precission, recall, accuracy, false_positive_rate, true_positive_rate, dict_new = calculate_pre_recall_accuracy_fp(
        actual_value_column_name, predicted_column_value_name)
    #
    # # print(precission)
    # # print(recall)
    # # print(accuracy)
    #
    # from sklearn import metrics
    # import matplotlib.pyplot as plt
    #
    # auc = metrics.roc_auc_score(actual_value_column_name, predicted_column_value_name)
    # # false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(actual_value_column_name, predicted_column_value_name)
    # plt.figure(figsize=(10, 8), dpi=100)
    # plt.axis('scaled')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.title("AUC & ROC Curve")
    # plt.plot(false_positive_rate, true_positive_rate, 'g')
    # plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
    # plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.show()

    # calculate_pri_recall(
    #     "/home/mahdiislam/Mahdi/Biba/iris-parcel-orientation-detection/Evalouation/ProcessedData/preprocessed_gt_orientation_result_6612.csv",
    #     "arrow_direction_gt_1",
    #     "arrow_direction_encode")



    import matplotlib.pyplot as plt
    from sklearn import svm, datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    from sklearn.multiclass import OneVsRestClassifier
    from itertools import cycle
    plt.style.use('ggplot')

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Binarize the output
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                     random_state=0))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(['blue', 'red', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")
    plt.show()