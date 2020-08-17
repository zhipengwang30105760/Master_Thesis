from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


from main_program import GENERATE_CONFUSION_MATRIX



#find the correlation value and drop the value which is larger than 0.34?
def correlation_heatmap(train, upper_bound, lower_bound):
    correlations = train.corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=0.1, annot=True, cbar_kws={"shrink": 0.7})
    #plt.show();
    columns = np.full((correlations.shape[0],), True, dtype=bool)
    for i in range(correlations.shape[0]):
        for j in range(i+1, correlations.shape[0]):
            if correlations.iloc[i,j] >= upper_bound or correlations.iloc[i,j] <= lower_bound:
                if columns[j]:
                    columns[j] = False;
                #drop both feature
                # if(columns[i]):
                #     columns[i] = False;

    selected_columns = train.columns[columns]
    #print(selected_columns[0:].values)
    data = train[selected_columns]
    #print(data)
    return data, selected_columns



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
                    columns = np.delete(columns, j)

    regressor_OLS.summary()
    return x, columns





if __name__ == "__main__":
    filename = r"/Users/zhipengwang/PycharmProjects/UNMC_Data_Analysis/data/original_Readmission_1.csv"
    target="Readmission_1"
    noisy_features = ['AGE', 'SMOKE', 'FNSTATUS2', 'HXCOPD', 'DIALYSIS', 'TRANSFUS', 'radial_all_yn', 'PRSEPIS']
    origin_data = pd.read_csv(filename)
    data, selected_columns = correlation_heatmap(origin_data, upper_bound= 0.25, lower_bound= -0.2)

    selected_columns = selected_columns[1:].values

    #decide whether use p-value verification or not
    #SL = 0.05
    # data_modeled, selected_columns = backwardElimination(data.iloc[:, 1:].values, data.iloc[:, 0].values, SL, selected_columns)
    #
    # result = pd.DataFrame()
    # result['diagnosis'] = data.iloc[:, 0]
    # data = pd.DataFrame(data=data_modeled, columns=selected_columns)
    selected_columns = list(selected_columns)
    for features in selected_columns:
        if features in noisy_features or features == target:
            selected_columns.remove(features)

    #X = origin_data.drop([target], axis=1)
    #X = data

    X= origin_data.loc[:, selected_columns]

    y = origin_data[target]
    r = GENERATE_CONFUSION_MATRIX(X, y, 0.2, target)
    score = r[0][0] + r[0][1] + r[1][0] + r[1][1]
    score = (r[0][0] + r[1][1]) / score

    print('Original result for selected correlated features without permutation')
    print(selected_columns)
    print(r)
    print(score)


    print('============================Get permutation list========================')
    comb = combinations(selected_columns, 10)
    print('============================Start Calculating===========================')
    i=1
    # print('number of all combination ' + str(sum( 1 for i in comb)))
    for combination in comb:
        print('========================Now at ' + str(i) + '==================')
        i += 1
        X = origin_data.loc[:, list(combination)]
        y = origin_data[target]
        curr_r = GENERATE_CONFUSION_MATRIX(X, y, 0.2, target)
        curr_score = curr_r[0][0] + curr_r[0][1] + curr_r[1][0] + curr_r[1][1]
        curr_score = (curr_r[0][0] + curr_r[1][1]) / curr_score
        if curr_score > score:
            print('============================Find Good Score===========================')
            print(list(combination))
            print(curr_r)
            print(curr_score)
            print('============================Keep Going===========================')
    print('==============================Finished===========================')