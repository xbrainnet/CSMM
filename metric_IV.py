from sklearn import metrics
from sklearn import preprocessing
import numpy as np


def cal_metrics(labels, predict):
    True_predict = 0
    TP0, FN0, FP0 = 0, 0, 0
    TP1, FN1, FP1 = 0, 0, 0
    TP2, FN2, FP2 = 0, 0, 0
    TP3, FN3, FP3 = 0, 0, 0
    Precisions = [0, 0, 0, 0]
    Recalls = [0, 0, 0, 0]
    Len = len(labels)
    for i in range(Len):
        if (labels[i] == predict[i]):
            True_predict += 1

        if (labels[i] == 0):
            if (predict[i] == 0):
                TP0 += 1
            else:
                FN0 += 1
        elif (labels[i] == 1):
            if (predict[i] == 1):
                TP1 += 1
            else:
                FN1 += 1
        elif (labels[i] == 2):
            if (predict[i] == 2):
                TP2 += 1
            else:
                FN2 += 1
        else:
            if (predict[i] == 3):
                TP3 += 1
            else:
                FN3 += 1

        if ((predict[i] == 0) and (labels[i] != 0)):
            FP0 += 1
        if ((predict[i] == 1) and (labels[i] != 1)):
            FP1 += 1
        if ((predict[i] == 2) and (labels[i] != 2)):
            FP2 += 1
        if ((predict[i] == 3) and (labels[i] != 3)):
            FP3 += 1

    ACC = True_predict / Len
    if (TP0 + FP0 == 0):
        TP0_FP0 = 0.00001
    else:
        TP0_FP0 = TP0 + FP0
    Precisions[0] = TP0 / (TP0_FP0)

    if (TP1 + FP1 == 0):
        TP1_FP1 = 0.00001
    else:
        TP1_FP1 = TP1 + FP1
    Precisions[1] = TP1 / (TP1_FP1)

    if (TP2 + FP2 == 0):
        TP2_FP2 = 0.00001
    else:
        TP2_FP2 = TP2 + FP2
    Precisions[2] = TP2 / (TP2_FP2)

    if (TP3 + FP3 == 0):
        TP3_FP3 = 0.00001
    else:
        TP3_FP3 = TP3 + FP3
    Precisions[3] = TP3 / (TP3_FP3)

    TP0_FN0 = TP0 + FN0
    TP1_FN1 = TP1 + FN1
    TP2_FN2 = TP2 + FN2
    TP3_FN3 = TP3 + FN3

    if (TP0_FN0 == 0):
        TP0_FN0 = 0.001
    if (TP1_FN1 == 0):
        TP1_FN1 = 0.001
    if (TP2_FN2 == 0):
        TP2_FN2 = 0.001
    if (TP3_FN3 == 0):
        TP3_FN3 = 0.001

    Recalls[0] = TP0 / (TP0_FN0)
    Recalls[1] = TP1 / (TP1_FN1)
    Recalls[2] = TP2 / (TP2_FN2)
    Recalls[3] = TP3 / (TP3_FN3)
    Precision_macro = 0.00001 + (Precisions[0] + Precisions[1] + Precisions[2] + Precisions[3]) / 4
    Recalls_macro = 0.00001 + (Recalls[0] + Recalls[1] + Recalls[2] + Recalls[3]) / 4

    macro_F1 = (2 * Precision_macro * Recalls_macro) / (Precision_macro + Recalls_macro)

    enc = preprocessing.OneHotEncoder()
    enc.fit([[0], [1], [2], [3]])

    labels = np.array(labels)
    predict = np.array(predict)

    labels = enc.transform(labels.reshape(-1, 1)).toarray()
    predict = enc.transform(predict.reshape(-1, 1)).toarray()

    AUC = metrics.roc_auc_score(labels, predict, average="macro", multi_class='ovr')

    return ACC, macro_F1, AUC, Precisions[0], Recalls[0], Precisions[1], Recalls[1], Precisions[2], Recalls[2], \
    Precisions[3], Recalls[3]
