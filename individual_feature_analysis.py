import itertools

import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import math
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import tensorflow as tf

import numpy as np

def drop_constant_columns(dataframe, candidate_features):
    """
    Drops constant value columns of pandas dataframe.
    """
    candidates_features = candidate_features.copy()
    result = dataframe.copy()
    for column in dataframe.columns:
        if len(dataframe[column].unique()) == 1:
            result = result.drop(column,axis=1)
            if column in feature_collections:
                feature_collections.remove(column)
    return result, candidates_features

def generate_classifier(layer, activation, solver, alpha, learning_rate):
    classifier = MLPClassifier(hidden_layer_sizes=layer, max_iter=500, activation=activation, solver=solver,
                               random_state=1, alpha=alpha, learning_rate=learning_rate)
    return classifier

def hba(matrix):
    tn = matrix[0][0]
    fp = matrix[0][1]
    fn = matrix[1][0]
    tp = matrix[1][1]

    score = (2*tp*tn) / (2*tp*tn + fp*tp + tn*fn)
    return score

def gba(matrix):
    tn = matrix[0][0]

    fp = matrix[0][1]
    fn = matrix[1][0]
    tp = matrix[1][1]

    sum = (tp / (tp + fn)) * (tn / (tn + fp))
    score = math.sqrt(sum)
    return score

#p is target, q is predict result
def cross_entropy(p, q):
    return -sum([p[i] * math.log2(q[i]) for i in range(len(p))])

def kl_divergence(p, q):
    return sum([p[i] * math.log2(p[i] / q[i]) for i in range(len(p))])


def individual_confusion_matrix():
    for feature in rest_features:
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(len(X)):
            if X[feature][i] == 1 and X[target][i] == 1:
                tp += 1
            elif X[feature][i] == 0 and X[target][i] == 1:
                fn += 1
            elif X[feature][i] == 1 and X[target][i] == 0:
                fp += 1
            elif X[feature][i] == 0 and X[target][i] == 0:
                tn += 1
        print(feature)
        confusion_matrix = [[tn, fp], [fn, tp]]
        print(confusion_matrix)

if __name__ == "__main__":
    filename = r"/Users/zhipengwang/PycharmProjects/UNMC_Data_Analysis/data/original_Reoperation_1.csv"
    target="Reoperation_1"
    # feature_collections = ['Groups', 'SEX', 'AGE', 'BMI', 'SMOKE', 'DYSPNEA', 'FNSTATUS2', 'HXCOPD', 'ASCITES', 'HXCHF',
    #                        'HYPERMED', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDIS', 'TRANSFUS',
    #                        'PRSEPIS', 'ASACLAS', 'radial_all_yn', 'distal_all_yn', 'race_final', 'Emerg_yn',
    #                        'Diabetes_yn', 'Pre_staging', 'PATHO_staging']
    feature_collections = ['Groups','ASACLAS','Pre_staging','PATHO_staging','AGE','BMI','DYSPNEA','FNSTATUS2',
                           'PRSEPIS']

    rest_features = ['SEX', 'SMOKE', 'HXCOPD', 'ASCITES', 'HXCHF',
                           'HYPERMED', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDIS', 'TRANSFUS', 'radial_all_yn', 'distal_all_yn', 'race_final', 'Emerg_yn',
                           'Diabetes_yn']

    data = pd.read_csv(filename)
    data, candidates_features = drop_constant_columns(data, feature_collections)
    y = data[target]
    X = data.drop(feature_collections, axis=1)


    output1 = []
    output2 = []

    target_positive = 0.05
    target_negative = 0.95
    p = []
    for i in range(len(X)):
        if X[target][i] == 1:
            p.append(target_positive)
        elif X[target][i] == 0:
            p.append(target_negative)


    for feature in rest_features:
        count = 0;
        for i in range(len(X)):
            if X[feature][i] == 1:
                count += 1
        feature_positive = count / len(X)
        feature_negative = 1- count / len(X)
        q = []
        for i in range(len(X)):
            if X[feature][i] == 1:
                q.append(feature_positive)
            elif X[feature][i] == 0:
                q.append(feature_negative)


        result1 = cross_entropy(p,q)
        result2 = kl_divergence(p, q)
        output1.append(result1)
        output2.append(result2)





    # do normalization for BMI and AGE
    #scaler = StandardScaler()
    #X.iloc[:, 2:4] = scaler.fit_transform(X.iloc[:, 2:4])


    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)


    # clf = svm.SVC(kernel='linear', C=1, probability=True)
    # clf = RandomForestClassifier(n_estimators=100)
    # clf.fit(X_train, y_train)


    #using proba function to get perentage of prediction, then can do cross entropy
    #prediction = mlp.predict_proba(X_test)
    # clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    # clf.fit(X, y)
    # clf = svm.SVC(kernel='linear', C=1)
    # clf.fit(X, y)

    # y_pred = clf.predict(X_test)
    # cm = confusion_matrix(y_pred, y_test)
    # print(cm)
    #
    # prediction = clf.predict_proba(X_test)
    # selected_feature_distribution = [predict for predict in prediction[:, 0]]
    # target_distribution = [score for score in y_test]
    # entropy = cross_entropy(target_distribution, selected_feature_distribution)
    # print(entropy)

    # final = [output1, output2]
    # df = pd.DataFrame(final, columns=feature_collections)
    # df.to_excel(r"/Users/zhipengwang/Desktop/output_result.xlsx", index=False)








