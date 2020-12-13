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
        confusion_matrix_list = [[[95989, 1288], [2203, 0]], [[97276, 1], [2203, 0]], [[97276, 1], [2203, 0]], [[96735, 542], [9, 2194]], [[97267, 5], [2203, 0]], [[27339, 69938], [0, 2203]], [[97209, 68], [74, 2129]], [[97254, 23], [2203, 0]], [[97266, 11], [2203, 0]], [[96707, 570], [2203, 0]], [[97044, 233], [2203, 0]], [[97234, 43], [2203, 0]], [[96833, 444], [2203, 0]], [[97277, 0], [2203, 0]], [[96906, 371], [2203, 0]], [[97162, 115], [2195, 8]], [[97180, 97], [2195, 8]], [[91841, 5436], [2140, 63]], [[91850, 5427], [2067, 136]], [[987, 96290], [0, 2203]], [[95027, 2250], [2197, 6]], [[86862, 10415], [1944, 259]], [[13635, 83642], [0, 2203]], [[91682, 5595], [2203, 0]], [[86412, 10865], [2195, 8]], [[97169, 108], [2203, 0]], [[97156, 121], [2203, 0]], [[97246, 31], [2203, 0]], [[91773, 5504], [2191, 12]], [[91608, 5669], [2191, 12]]]
        confusion_matrix_list = [[[1,2],[3,4]]]
        positive = []
        negative = []
        scores = []
        for matrix in confusion_matrix_list:
            # tpr, tnr = return_tpr_tnr(matrix)
            score = tpr_times_tnr(matrix)
            scores.append(score)


        final = [scores]

        df = pd.DataFrame(final, columns=binary_feature_collections)
        df.to_excel(r"/Users/zhipengwang/Desktop/output_result.xlsx", index=False)
