import pandas as pd
from sklearn.neural_network import MLPClassifier
import math

def tpr_times_tnr(matrix):
    tn = matrix[0][0]
    fn = matrix[0][1]
    fp = matrix[1][0]
    tp = matrix[1][1]

    try:
        score = (tp / (tp + fn)) * (tn / (tn + fp))
    except:
        score = 1
    return score

def return_tpr_tnr(matrix):
    tn = matrix[0][0]
    fp = matrix[1][0]
    fn = matrix[0][1]
    tp = matrix[1][1]

    if tp + fn == 0:
        tpr = 1
    else:
        tpr = tp / (tp + fn)
    if tn + fp == 0:
        tnr = 1
    else:
        tnr = tn / (tn + fp)

    return tpr, tnr


if __name__ == "__main__":
    binary_feature_collections = ['CM_AIDS','CM_ALCOHOL','CM_ANEMDEF','CM_ARTH','CM_BLDLOSS','CM_CHF','CM_CHRNLUNG','CM_COAG','CM_DEPRESS','CM_DM'
        ,'CM_DMCX','CM_DRUG','CM_HTN_C','CM_HYPOTHY','CM_LIVER','CM_LYMPH','CM_LYTES','CM_METS','CM_NEURO','CM_OBESE','CM_PARA','CM_PERIVASC','CM_PSYCH'
        ,'CM_PULMCIRC','CM_RENLFAIL','CM_TUMOR','CM_ULCER','CM_VALVE','CM_WGHTLOSS']
    confusion_matrix_list = [[[67944, 55], [2439, 2]], [[66302, 1697], [2368, 73]], [[47168, 20831], [1735, 706]], [[66727, 1272], [2403, 38]], [[63255, 4744], [2296, 145]], [[62122, 5877], [1780, 661]], [[56495, 11504], [1881, 560]], [[65758, 2241], [1967, 474]], [[63031, 4968], [2309, 132]], [[53409, 14590], [1997, 444]], [[66040, 1959], [2355, 86]], [[67467, 532], [2420, 21]], [[26049, 41950], [1143, 1298]], [[60451, 7548], [2205, 236]], [[66386, 1613], [2328, 113]], [[67562, 437], [2413, 28]], [[48437, 19562], [894, 1547]], [[44103, 23896], [1467, 974]], [[64725, 3274], [2229, 212]], [[59835, 8164], [2255, 186]], [[67127, 872], [2353, 88]], [[64369, 3630], [2093, 348]], [[66278, 1721], [2359, 82]], [[66031, 1968], [2225, 216]], [[63011, 4988], [1974, 467]], [[63131, 4868], [2092, 349]], [[67967, 32], [2441, 0]], [[64430, 3569], [2248, 193]], [[59694, 8305], [1647, 794]]]

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
