import pandas as pd
if __name__ == "__main__":
    target = "Approach"
    binary_feature_collections = ['FEMALE', 'CM_AIDS', 'CM_ALCOHOL', 'CM_ANEMDEF', 'CM_ARTH', 'CM_BLDLOSS', 'CM_CHF',
                                  'CM_CHRNLUNG', 'CM_COAG', 'CM_DEPRESS', 'CM_DM', 'CM_DMCX', 'CM_DRUG', 'CM_HTN_C',
                                  'CM_HYPOTHY', 'CM_LIVER', 'CM_LYMPH', 'CM_LYTES', 'CM_METS', 'CM_NEURO',
                                  'CM_OBESE', 'CM_PARA', 'CM_PERIVASC', 'CM_PSYCH', 'CM_PULMCIRC', 'CM_RENLFAIL',
                                  'CM_TUMOR', 'CM_ULCER', 'CM_VALVE', 'CM_WGHTLOSS', 'DIED']
    filename = r"/Users/zhipengwang/PycharmProjects/UNMC_Data_Analysis/data/single_Approach_analyze.csv"
    data = pd.read_csv(filename, encoding='ISO-8859-1')

    """
    here is the previous formula
    ((number of case whose value is 1 and target is 1) / (total number of target whose value is 1))
    /
    ((number of case whose value is 1 and target is 0) / (total number of target whose value is 0))
    is more than 2.0

    or 
    
    ((number of case whose value is 1 and target is 0) / (total number of target whose value is 0))
    /
    ((number of case whose value is 1 and target is 1) / (total number of target whose value is 1))
    is more than 2.0
    
    """
    total_one = 123
    total_zero = 877
    score_list = []
    for feature in binary_feature_collections:
        one_and_one = 0
        one_and_zero = 0
        for i in range(len(data)):
            if data[feature][i] == 1 and data[target][i] == 1:
                one_and_one += 1
            elif data[feature][i] == 1 and data[target][i] == 0:
                one_and_zero += 1

        score1 = 0
        score2 = 0
        if (one_and_zero / total_zero) != 0:
            score1 = (one_and_one / total_one) / (one_and_zero / total_zero)

        if (one_and_one / total_one) != 0:
            score2 = (one_and_zero / total_zero) / (one_and_one / total_one)

        if score1 > 2.0 or score2 > 2.0:
            score_list.append(feature)



    print(score_list)
