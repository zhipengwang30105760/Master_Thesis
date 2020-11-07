import pandas as pd
from sklearn.neural_network import MLPClassifier
import math

def tpr_times_tnr(matrix):
    tn = matrix[0][0]
    fp = matrix[0][1]
    fn = matrix[1][0]
    tp = matrix[1][1]

    score = (tp / (tp + fn)) * (tn / (tn + fp))
    return score

def return_tpr_tnr(matrix):
    tn = matrix[0][0]
    fp = matrix[0][1]
    fn = matrix[1][0]
    tp = matrix[1][1]

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    return tpr, tnr


if __name__ == "__main__":
    binary_feature_collections = ['FEMALE','CM_AIDS','CM_ALCOHOL','CM_ANEMDEF','CM_ARTH','CM_BLDLOSS','CM_CHF','CM_CHRNLUNG','CM_COAG','CM_DEPRESS','CM_DM','CM_DMCX','CM_DRUG','CM_HTN_C',
                           'CM_HYPOTHY','CM_LIVER','CM_LYMPH','CM_LYTES','CM_METS','CM_NEURO',
                           'CM_OBESE','CM_PARA','CM_PERIVASC','CM_PSYCH','CM_PULMCIRC','CM_RENLFAIL','CM_TUMOR','CM_ULCER','CM_VALVE','CM_WGHTLOSS']
    confusion_matrix_list = [[[488, 447], [29, 36]],	[[933, 2], [65, 0]],	[[892, 43], [63, 2]]	,[[603, 332], [44, 21]]	,[[913, 22], [64, 1]]	,[[876, 59], [63, 2]]	,[[823, 112], [49, 16]]	,[[735, 200], [48, 17]]	,[[877, 58], [46, 19]],	[[863, 72], [62, 3]]	,[[734, 201], [52, 13]]	,[[905, 30], [64, 1]],	[[927, 8], [65, 0]]	,[[366, 569], [32, 33]],	[[847, 88], [59, 6]],	[[907, 28], [61, 4]]	,[[924, 11], [64, 1]],	[[542, 393], [20, 45]],	[[589, 346], [37, 28]],	[[885, 50], [60, 5]]	,[[832, 103], [58, 7]]	,[[920, 15], [60, 5]]	,[[876, 59], [60, 5]]	,[[906, 29], [62, 3]]	,[[882, 53], [58, 7]],	[[861, 74], [52, 13]],	[[859, 76], [61, 4]],	[[935, 0], [65, 0]]	,[[879, 56], [59, 6]],	[[747, 188], [38, 27]]]

    positive = []
    negative = []
    scores = []
    for matrix in confusion_matrix_list:
        tpr, tnr = return_tpr_tnr(matrix)
        score = tpr_times_tnr(matrix)
        scores.append(score)
        positive.append(tpr)
        negative.append(tnr)


    final = [positive, negative, scores]

    df = pd.DataFrame(final, columns=binary_feature_collections)
    df.to_excel(r"/Users/zhipengwang/Desktop/output_result.xlsx", index=False)
