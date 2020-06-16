import math

def matrix_position(matrix, cls):
    if cls == 0:
        TP = matrix[0][0]
        FN = matrix[0][1]
        FP = matrix[1][0]
        TN = matrix[1][1]
    elif cls == 1:
        TN = matrix[0][0]
        FP = matrix[0][1]
        FN = matrix[1][0]
        TP = matrix[1][1]
    return TP, FN, FP, TN
def recall(tp, fn):
    score = tp / (tp + fn)
    return score
def precision(tp, fp):
    if tp + fp != 0:
        score = tp / (tp + fp)
    else:
        score = 100
    return score
def g_mean(tp, fp, tn, fn):
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    gmean = math.sqrt(tpr * tnr)
    return gmean
if __name__ == "__main__":
    matrix = [[291, 3], [60, 0]]
    TP1, FN1, FP1, TN1 = matrix_position(matrix, 1)
    recall_score1 = recall(TP1, FN1)
    precision_score1 = precision(TP1, FP1)
    gmean_score1 = g_mean(TP1, FP1, TN1, FN1)
    print("result for class 1")
    print("recall1 is:  %.2f" % recall_score1)
    print("precision1 is:  %.2f" % precision_score1)
    print("gmean1 is:  %.2f" % gmean_score1)
    print()


    tp0, fn0, fp0, tn0 = matrix_position(matrix, 0)
    recall_score0 = recall(tp0, fn0)
    precision_score0 = precision(tp0, fp0)
    gmean_score0 = g_mean(tp0, fp0, tn0, fn0)
    print("result for class 0")
    print("recall0 is:  %.2f" % recall_score0)
    print("precision0 is:  %.2f" % precision_score0)
    print("gmean0 is:  %.2f" % gmean_score0)
    # print(TP1)
    # print(FN1)
    # print(FP1)
    # print(TN1)