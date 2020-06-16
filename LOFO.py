import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from pandas import read_csv
from numpy import set_printoptions
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2, f_regression
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectPercentile, VarianceThreshold
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def decision_tree_backward(X, y, n_selected_features):
    """
    This function implements the backward feature selection algorithm based on decision tree

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels
    n_selected_features : {int}
        number of selected features

    Output
    ------
    F: {numpy array}, shape (n_features, )
        index of selected features
    """

    n_samples, n_features = X.shape
    # using 10 fold cross validation
    #cv = KFold(n_samples, n_folds=10, shuffle=True)
    cv = KFold(n_samples, shuffle=True)
    # choose decision tree as the classifier
    clf = DecisionTreeClassifier()

    # selected feature set, initialized to contain all features
    F = list(range(n_features))
    count = n_features

    while count > n_selected_features:
        max_acc = 0
        for i in range(n_features):
            if i in F:
                F.remove(i)
                X_tmp = X[:, F]
                acc = 0
                for train, test in cv.split(X):
                    clf.fit(X_tmp[train], y[train])
                    y_predict = clf.predict(X_tmp[test])
                    acc_tmp = accuracy_score(y[test], y_predict)
                    acc += acc_tmp
                acc = float(acc)/10
                F.append(i)
                # record the feature which results in the largest accuracy
                if acc > max_acc:
                    max_acc = acc
                    idx = i
        # delete the feature which results in the largest accuracy
        F.remove(idx)
        count -= 1
    return np.array(F)

def decision_tree_forward(X, y, n_selected_features):
    """
    This function implements the forward feature selection algorithm based on decision tree

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples, )
        input class labels
    n_selected_features: {int}
        number of selected features

    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features
    """

    n_samples, n_features = X.shape
    # using 10 fold cross validation
    cv = KFold(n_samples, shuffle=True)
    # choose decision tree as the classifier
    clf = DecisionTreeClassifier()

    # selected feature set, initialized to be empty
    F = []
    count = 0
    while count < n_selected_features:
        max_acc = 0
        for i in range(n_features):
            if i not in F:
                F.append(i)
                X_tmp = X[:, F]
                acc = 0
                for train, test in cv.split(X):
                    clf.fit(X_tmp[train], y[train])
                    y_predict = clf.predict(X_tmp[test])
                    acc_tmp = accuracy_score(y[test], y_predict)
                    acc += acc_tmp
                acc = float(acc)/10
                F.pop()
                # record the feature which results in the largest accuracy
                if acc > max_acc:
                    max_acc = acc
                    idx = i
        # add the feature which results in the largest accuracy
        F.append(idx)
        count += 1
    return np.array(F)
def svm_backward(X, y, n_selected_features):
    """
    This function implements the backward feature selection algorithm based on SVM

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels
    n_selected_features: {int}
        number of selected features

    Output
    ------
    F: {numpy array}, shape (n_features, )
        index of selected features
    """

    n_samples, n_features = X.shape
    # using 10 fold cross validation
    cv = KFold(n_samples, shuffle=True)
    # choose SVM as the classifier
    clf = SVC()

    # selected feature set, initialized to contain all features
    F = list(range(n_features))
    count = n_features

    while count > n_selected_features:
        max_acc = 0
        for i in range(n_features):
            if i in F:
                F.remove(i)
                X_tmp = X[:, F]
                acc = 0
                for train, test in cv.split(X):
                    clf.fit(X_tmp[train], y[train])
                    y_predict = clf.predict(X_tmp[test])
                    acc_tmp = accuracy_score(y[test], y_predict)
                    acc += acc_tmp
                acc = float(acc)/10
                F.append(i)
                # record the feature which results in the largest accuracy
                if acc > max_acc:
                    max_acc = acc
                    idx = i
        # delete the feature which results in the largest accuracy
        F.remove(idx)
        count -= 1
    return np.array(F)

def svm_forward(X, y, n_selected_features):
    """
    This function implements the forward feature selection algorithm based on SVM

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels
    n_selected_features: {int}
        number of selected features

    Output
    ------
    F: {numpy array}, shape (n_features, )
        index of selected features
    """

    n_samples, n_features = X.shape
    # using 10 fold cross validation
    cv = KFold(n_samples, shuffle=True)
    # choose SVM as the classifier
    clf = SVC()

    # selected feature set, initialized to be empty
    F = []
    count = 0
    while count < n_selected_features:
        max_acc = 0
        for i in range(n_features):
            if i not in F:
                F.append(i)
                X_tmp = X[:, F]
                acc = 0
                for train, test in cv.split(X):
                    clf.fit(X_tmp[train], y[train])
                    y_predict = clf.predict(X_tmp[test])
                    acc_tmp = accuracy_score(y[test], y_predict)
                    acc += acc_tmp
                acc = float(acc)/10
                F.pop()
                # record the feature which results in the largest accuracy
                if acc > max_acc:
                    max_acc = acc
                    idx = i
        # add the feature which results in the largest accuracy
        F.append(idx)
        count += 1
    return np.array(F)

def convert_Index_to_Name(names, result):
    name_list = []
    for i in result:
        name_list.append(names[i])
    #print(name_list)
    return name_list

if __name__ == "__main__":    
    # load data
    filename = r"C:\Users\zhipe\Desktop\target3.csv"
    names = ['Groups', 'SEX', 'AGE', 'BMI', 'SMOKE', 'DYSPNEA', 'FNSTATUS2', 'HXCOPD', 'ASCITES', 'HXCHF', 'HYPERMED', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDIS', 'TRANSFUS', 'PRSEPIS', 'ASACLAS', 'radial_all_yn', 'distal_all_yn', 'race_final', 'Emerg_yn', 'Diabetes_yn', 'Pre_staging', 'PATHO_staging', 'Reoperation_1']
    dataframe = read_csv(filename)
    array = dataframe.values
    for i in range(len(array)):
        for j in range(len(array[i])):
            value = float(array[i][j])
            array[i][j] = value
    #the last one is target
    X = array[:,0:27]
    Y = array[:,27]
    print('==============first start working===========')
    bk1 = decision_tree_backward(X, Y, 25)
    result1 = convert_Index_to_Name(names, bk1)
    print(result1)
    print('==============first finished===========')

    print('==============second start working===========')
    bk2 = decision_tree_forward(X, Y, 25)
    result2 = convert_Index_to_Name(names, bk2)
    print(result2)
    print('==============second finished===========')

    print('==============third start working===========')
    fw1 = svm_forward(X, Y, 25)
    result3 = convert_Index_to_Name(names, fw1)
    print(result3)
    print('==============third finished===========')


    fw2 = svm_backward(X, Y, 25)
    result4 = convert_Index_to_Name(names, fw2)
    print(result4)
    #for 20
    bk3 = decision_tree_backward(X, Y, 20)
    result5 = convert_Index_to_Name(names, bk3)
    print(result5)
    bk4 = decision_tree_forward(X, Y, 20)
    result6 = convert_Index_to_Name(names, bk4)
    print(result6)
    fw3 = svm_forward(X, Y, 20)
    result7 = convert_Index_to_Name(names, fw3)
    print(result7)
    fw4 = svm_backward(X, Y, 20)
    result8 = convert_Index_to_Name(names, fw4)
    print(result8)
    sum = [result1, result2, result3, result4, result5, result6, result7, result8]
    print(sum)
    df = pd.DataFrame(sum)
    df.to_excel(r"C:\Users\zhipe\Desktop\result1.xlsx", index = False)
