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
            if column in candidates_features:
                candidates_features.remove(column)
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


def individual_confusion_matrix(X, non_binary_feature_collections, target):
    result = []
    for feature in non_binary_feature_collections:
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
        # print(feature)
        confusion_matrix = [[tn, fp], [fn, tp]]
        result.append(confusion_matrix)
        # print(confusion_matrix)
    return result

def prev_feature_set():
    feature_collections = ['Groups', 'SEX', 'AGE', 'BMI', 'SMOKE', 'DYSPNEA', 'FNSTATUS2', 'HXCOPD', 'ASCITES', 'HXCHF',
                           'HYPERMED', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDIS', 'TRANSFUS',
                           'PRSEPIS', 'ASACLAS', 'radial_all_yn', 'distal_all_yn', 'race_final', 'Emerg_yn',
                           'Diabetes_yn', 'Pre_staging', 'PATHO_staging']
    #reoperation, readmission, mortality
    non_binary_feature_collections = ['Groups','ASACLAS','Pre_staging','PATHO_staging','AGE','BMI','DYSPNEA','FNSTATUS2',
                           'PRSEPIS']

    binary_feature_collections = ['SEX', 'SMOKE', 'HXCOPD', 'ASCITES', 'HXCHF',
                           'HYPERMED', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDIS', 'TRANSFUS', 'radial_all_yn', 'distal_all_yn', 'race_final', 'Emerg_yn',
                           'Diabetes_yn']
    #approach died
    binary_feature_collections = ['FEMALE', 'CM_AIDS', 'CM_ALCOHOL', 'CM_ANEMDEF', 'CM_ARTH', 'CM_BLDLOSS', 'CM_CHF',
                                  'CM_CHRNLUNG', 'CM_COAG', 'CM_DEPRESS', 'CM_DM', 'CM_DMCX', 'CM_DRUG', 'CM_HTN_C',
                                  'CM_HYPOTHY', 'CM_LIVER', 'CM_LYMPH', 'CM_LYTES', 'CM_METS', 'CM_NEURO',
                                  'CM_OBESE', 'CM_PARA', 'CM_PERIVASC', 'CM_PSYCH', 'CM_PULMCIRC', 'CM_RENLFAIL',
                                  'CM_TUMOR', 'CM_ULCER', 'CM_VALVE', 'CM_WGHTLOSS']
if __name__ == "__main__":
    filename = r"/Users/zhipengwang/PycharmProjects/UNMC_Data_Analysis/data/kddcup99.csv"
    target="label"

    binary_feature_collections = ['protocol_type','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','lnum_compromised',
                                  'lroot_shell','lsu_attempted','lnum_root','lnum_file_creations','lnum_shells','lnum_access_files','lnum_outbound_cmds',
                                  'is_host_login','is_guest_login','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
                                  'same_srv_rate','diff_srv_rate','srv_diff_host_rate',	'dst_host_same_srv_rate','dst_host_diff_srv_rate',
                                  'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
                                  'dst_host_rerror_rate','dst_host_srv_rerror_rate']
    binary_feature_collections.remove('wrong_fragment')
    binary_feature_collections.remove('lnum_outbound_cmds')

    data = pd.read_csv(filename, encoding='ISO-8859-1')
    X = data
    # data, candidates_features = drop_constant_columns(data, feature_collections)
    result = individual_confusion_matrix(X, binary_feature_collections, target)
    print(result)




    #y = data[target]
    #X = data.drop(feature_collections, axis=1)


    # output1 = []
    # output2 = []
    #
    # target_positive = 0.065
    # target_negative = 0.935
    # p = []
    # for i in range(len(X)):
    #     if X[target][i] == 1:
    #         p.append(target_positive)
    #     elif X[target][i] == 0:
    #         p.append(target_negative)
    #
    # count = 0
    # for i in range(len(X)):
    #     if X["CM_DRUG"][i] == 1:
    #         count = count + 1
    # feature_positive = count / len(X)
    # feature_negative = 1 - count / len(X)
    # q= []
    # for i in range(len(X)):
    #     if X["CM_DRUG"][i] == 1:
    #         q.append(feature_positive)
    #     elif X["CM_DRUG"][i] == 0:
    #         q.append(feature_negative)
    #
    # print(p)
    # print(q)


    # for feature in binary_feature_collections:
    #     count = 0;
    #     for i in range(len(X)):
    #         if X[feature][i] == 1:
    #             count += 1
    #     feature_positive = count / len(X)
    #     feature_negative = 1- count / len(X)
    #     q = []
    #     for i in range(len(X)):
    #         if X[feature][i] == 1:
    #             q.append(feature_positive)
    #         elif X[feature][i] == 0:
    #             q.append(feature_negative)
    #
    #
    #     result1 = cross_entropy(p,q)
    #     result2 = kl_divergence(p, q)
    #     output1.append(result1)
    #     output2.append(result2)
    #
    #
    #
    #
    # final = [output1, output2]
    #
    df = pd.DataFrame(result, columns=['first', 'second'])
    df.to_excel(r"/Users/zhipengwang/Desktop/output_result.xlsx", index=False)







