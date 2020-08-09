import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from main_program import GENERATE_CONFUSION_MATRIX



#find the correlation value and drop the value which is larger than 0.34?
def correlation_heatmap(train):
    correlations = train.corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=0.1, annot=True, cbar_kws={"shrink": 0.7})
    #plt.show();
    columns = np.full((correlations.shape[0],), True, dtype=bool)
    for i in range(correlations.shape[0]):
        for j in range(i+1, correlations.shape[0]):
            if correlations.iloc[i,j] >=0.9:
                if columns[j]:
                    columns[j] = False;

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
    filename = r"/Users/zhipengwang/PycharmProjects/UNMC_Data_Analysis/data/oversampling_Readmission_1.csv"
    target="Readmission_1"
    origin_data = pd.read_csv(filename)
    data, selected_columns = correlation_heatmap(origin_data)

    selected_columns = selected_columns[1:].values

    SL = 0.05

    data_modeled, selected_columns = backwardElimination(data.iloc[:, 1:].values, data.iloc[:, 0].values, SL, selected_columns)

    result = pd.DataFrame()
    result['diagnosis'] = data.iloc[:, 0]
    data = pd.DataFrame(data=data_modeled, columns=selected_columns)

    print(selected_columns)
    #X = origin_data.drop([target], axis=1)
    X = data
    if target in selected_columns:
        X = X.drop([target], axis=1)

    y = origin_data[target]
    r = GENERATE_CONFUSION_MATRIX(X, y, 0.2, target)
    print(r)

