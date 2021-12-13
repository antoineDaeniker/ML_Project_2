import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (PrecisionRecallDisplay, accuracy_score, auc,
                             average_precision_score, confusion_matrix,
                             fbeta_score, precision_recall_curve,
                             precision_score, recall_score, f1_score, roc_curve)


def metrics_from_confusion_matrix(_confusion_matrix):
    TN, FN, TP, FP = _confusion_matrix[np.array([0, 1, 1, 0]), np.array([0, 0, 1, 1])]
    acc = (TP + TN) / (TN + FN + TP + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    _f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    IoU = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    return {"acc": acc, "precision": precision, "recall": recall, "f1_score": _f1_score, "IoU":IoU}


def metrics(y_pred, y_true, threshold=0.5, curve=False):

    ######################################
    ###### NON THRESHOLDED METRICS #######
    ######################################

    # ROC CURVE (and AUC)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_ROC_score = auc(fpr, tpr)

    # PRECISION-RECALL CURVE (and AUC)
    precision_values, recall_values, _ = precision_recall_curve(y_true, y_pred)
    auc_PRC_score = auc(recall_values, precision_values)

    # AVERAGE PRECISION SCORE
    AP_score = average_precision_score(y_true, y_pred)

    ######################################
    ######## THRESHOLDED METRICS #########
    ######################################

    y_pred_thresholded = (y_pred>threshold).type(torch.uint8)


    # ACCURACY, PRECISION, RECALL
    acc = accuracy_score(y_true, y_pred_thresholded)
    precision = precision_score(y_true, y_pred_thresholded)
    recall = recall_score(y_true, y_pred_thresholded)
    _f1_score = f1_score(y_true, y_pred_thresholded)

    if curve:
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Area = {:.2f}".format(auc_ROC_score))
        plt.show()

        disp = PrecisionRecallDisplay(precision=precision_values, recall=recall_values)
        disp.plot(label = "AP: {}".format(AP_score))
        plt.title("Precision-Recall Curve - Area = {:.2f}".format(auc_PRC_score))
        plt.legend()
        plt.show()

    return {"acc": acc, "precision": precision, "recall": recall, "f1_score": _f1_score, "AP_score": AP_score, "AUC_PRC": auc_PRC_score, "AUC_ROC": auc_ROC_score}