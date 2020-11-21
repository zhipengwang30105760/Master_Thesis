import pandas as pd
from sklearn.neural_network import MLPClassifier
import math
class TPR_and_TNR:
    def __init__(self, name=None):
        self.name = name

    def tpr_times_tnr(self, matrix):
        tn = matrix[0][0]
        fp = matrix[0][1]
        fn = matrix[1][0]
        tp = matrix[1][1]

        try:
            score = (tp / (tp + fn)) * (tn / (tn + fp))
        except:
            score = 1
        return score

    def tpr_minus_fpr(self, matrix):
        tn = matrix[0][0]
        fp = matrix[0][1]
        fn = matrix[1][0]
        tp = matrix[1][1]

        try:
            score = (tp / (tp + fn)) - (fp / (tn + fp))
        except:
            score = 1
        return score

    def return_tpr_tnr(matrix):
        tn = matrix[0][0]
        fp = matrix[0][1]
        fn = matrix[1][0]
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
        confusion_matrix_list = [[[16988, 34], [582, 6]], [[16675, 347],[498, 90]], [[15875, 1147],[389, 199]], [[13575, 46],[450, 17]], [[13354, 267],[396, 71]], [[12709, 912],[310, 157]]
                                 , [[20108, 310],[638, 76]], [[20008, 410],[604, 110]], [[19035, 1383],[468, 246]]]

        positive = []
        negative = []
        scores = []
        for matrix in confusion_matrix_list:
            # tpr, tnr = return_tpr_tnr(matrix)
            score = tpr_minus_fpr(matrix)
            scores.append(score)


        final = [scores]

        df = pd.DataFrame(final, columns=binary_feature_collections)
        df.to_excel(r"/Users/zhipengwang/Desktop/output_result.xlsx", index=False)
