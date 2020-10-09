from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


from classification_using_normal_model import GENERATE_CONFUSION_MATRIX




def correlation_heatmap(train, upper_bound, lower_bound, feature_collections):
    dependent_features = [[]]
    correlations = train.corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=0.1, annot=True, cbar_kws={"shrink": 0.7})
    #plt.show();
    columns = np.full((correlations.shape[0],), True, dtype=bool)
    for i in range(correlations.shape[0]):
        for j in range(i+1, correlations.shape[0]):
            if correlations.iloc[i,j] >= upper_bound or correlations.iloc[i,j] <= lower_bound:
                # drop both feature
                if columns[j]:
                    columns[j] = False;
                if(columns[i]):
                    columns[i] = False;
                dependent_features.append([feature_collections[i], feature_collections[j]])

    independent_features = train.columns[columns]
    #print(selected_columns[0:].values)
    data = train[independent_features]
    #print(data)
    return data, independent_features, dependent_features



#do a verification based on our assumption
def backwardElimination(x, Y, sl, columns):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    ccolumns = np.delete(columns, j)

    regressor_OLS.summary()
    return x, columns

def getQk(name):
    if name == 'Readmission':
        return [0.022367692,0.011623414,0.010608173,0.088069519,0.003058197,0.107589514,0.102943274,0.016245358,
           0.122336971,0.502047782,0.058147723,0.085628626,0.047023973,0.391624511,0.066956472,
           0.05408992,0.170489244,0.027203207,0.003232422,0.088553901,0.006802413,0.001974265,0.037445756,0.102285681,0.115130069,0.017297456,0.039623768] #for readmission
    if name == 'Mortality':
        return[0.21,0.11,0.8,0.14,0.061,0.61,0.46,0.35,0.5,0.5,0.006,0.5,0.166,	0.5,0.3,0.2,0.51,0.44,0.37,0.37,0.166,0.02,0.26,0.023,0.19,0.09,0.09]

    if name == 'Reoperation':
        return [0.12,0.04,0.086,0.049,0.054,0.019,0.045,0.045,0.35,0.4,0.017,0.5,0.183,0.313,0.07,0.2,0.5,0.1,0.21,0.09,0.06,0.05,0.096,0.08,0.061,0.163,0.033]


if __name__ == "__main__":
    # filename = r"/Users/zhipengwang/PycharmProjects/UNMC_Data_Analysis/data/original_Readmission_1.csv"
    # target="Readmission_1"
    # origin_data = pd.read_csv(filename)
    feature_collections = ['Groups', 'SEX', 'AGE', 'BMI', 'SMOKE', 'DYSPNEA', 'FNSTATUS2', 'HXCOPD', 'ASCITES', 'HXCHF',
                           'HYPERMED', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDIS', 'TRANSFUS',
                           'PRSEPIS', 'ASACLAS', 'radial_all_yn', 'distal_all_yn', 'race_final', 'Emerg_yn',
                           'Diabetes_yn', 'Pre_staging', 'PATHO_staging']


    # data, independent_features, dependent_features = correlation_heatmap(origin_data, upper_bound= 0.2, lower_bound= -0.2, feature_collections = feature_collections)
    # dependent_features = dependent_features[1:] #first list is empty set we should remove it
    # print(dependent_features)
    #print(independent_features)

    Qk = getQk('Reoperation')

    candidate_list = [['AGE', 'HYPERMED'],
                      ['AGE', 'ASACLAS'],
                      ['BMI', 'HYPERMED'],
                      ['BMI', 'Diabetes_yn'],
                      ['DYSPNEA', 'HXCOPD'],
                      ['FNSTATUS2', 'ASCITES'],
                      ['HYPERMED', 'ASACLAS'],
                      ['HYPERMED', 'Diabetes_yn'],
                      ['DISCANCR', 'Pre_staging'],
                      ['DISCANCR', 'PATHO_staging'],
                      ['TRANSFUS', 'PRSEPIS'],
                      ['TRANSFUS', 'Emerg_yn'],
                      ['PRSEPIS', 'Emerg_yn']]

    for candidates in candidate_list:
        q1 = Qk[feature_collections.index(candidates[0])]
        q2 = Qk[feature_collections.index(candidates[1])]
        if q1 > q2:
            candidates.remove(candidates[0])
        else:
            candidates.remove(candidates[1])

    result = set()
    for candidates in candidate_list:
        result.add(candidates[0])

    print(result)

    #independent features ['Groups', 'SEX', 'SMOKE', 'HXCHF', 'DIALYSIS', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDIS', 'radial_all_yn', 'distal_all_yn', 'race_final']

