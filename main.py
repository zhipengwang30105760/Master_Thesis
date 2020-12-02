import pandas as pd


def tpr_minus_fpr(matrix):
    tn = matrix[0][0]
    fp = matrix[0][1]
    fn = matrix[1][0]
    tp = matrix[1][1]

    try:
        score = (tp / (tp + fn)) - (fp / (tn + fp))
    except:
        score = 1
    return score

def tpr_times_tnr(matrix):
    tn = matrix[0][0]
    fp = matrix[0][1]
    fn = matrix[1][0]
    tp = matrix[1][1]

    try:
        score = (tp / (tp + fn)) * (tn / (tn + fp))
    except:
        score = 1
    return score

if __name__ == "__main__":

    confusion_matrix_list = [[[60174, 50], [9909, 7]], [[58677, 1547], [9700, 216]], [[40945, 19279], [7725, 2191]], [[59097, 1127], [9734, 182]], [[55748, 4476], [9512, 404]], [[54271, 5953], [9346, 570]], [[49726, 10498], [8402, 1514]], [[57763, 2461], [9669, 247]], [[55857, 4367], [9217, 699]], [[47356, 12868], [7829, 2087]], [[58454, 1770], [9643, 273]], [[59733, 491], [9855, 61]], [[23526, 36698], [3574, 6342]], [[53557, 6667], [8836, 1080]], [[58700, 1524], [9719, 197]], [[59811, 413], [9866, 50]], [[41053, 19171], [8023, 1893]], [[37866, 22358], [7463, 2453]], [[57163, 3061], [9501, 415]], [[53374, 6850], [8458, 1458]], [[59360, 864], [9822, 94]], [[56687, 3537], [9486, 430]], [[58648, 1576], [9694, 222]], [[58240, 1984], [9721, 195]], [[55407, 4817], [9293, 623]], [[55594, 4630], [9352, 564]], [[60194, 30], [9914, 2]], [[56923, 3301], [9462, 454]], [[51788, 8436], [9263, 653]]]
    score_list = []
    for matrix in confusion_matrix_list:
        score = tpr_times_tnr(matrix)
        score_list.append(score)
    print(score_list)
    final = [score_list]

    binary_feature_collections = ['CM_AIDS', 'CM_ALCOHOL', 'CM_ANEMDEF', 'CM_ARTH', 'CM_BLDLOSS', 'CM_CHF',
                                  'CM_CHRNLUNG', 'CM_COAG', 'CM_DEPRESS', 'CM_DM', 'CM_DMCX', 'CM_DRUG', 'CM_HTN_C',
                                  'CM_HYPOTHY', 'CM_LIVER', 'CM_LYMPH', 'CM_LYTES', 'CM_METS', 'CM_NEURO',
                                  'CM_OBESE', 'CM_PARA', 'CM_PERIVASC', 'CM_PSYCH', 'CM_PULMCIRC', 'CM_RENLFAIL',
                                  'CM_TUMOR', 'CM_ULCER', 'CM_VALVE', 'CM_WGHTLOSS']
    df = pd.DataFrame(final, columns=binary_feature_collections)
    df.to_excel(r"/Users/zhipengwang/Desktop/output_result.xlsx", index=False)

