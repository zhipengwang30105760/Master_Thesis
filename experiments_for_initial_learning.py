from pandas import read_csv
from pandas.plotting import scatter_matrix
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn import pipeline
from sklearn import ensemble
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import math
def read_Data(filename):
    data = pd.read_csv(filename)
    return data

def g_Mean_class_0(matrix):
    TP = matrix[0][0]
    FN = matrix[0][1]
    FP = matrix[1][0]
    TN = matrix[1][1]
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    G_MEAN = math.sqrt(TPR * TNR)
    return G_MEAN
def g_Mean_class_1(matrix):
    TN = matrix[0][0]
    FP = matrix[0][1]
    FN = matrix[1][0]
    TP = matrix[1][1]
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    G_MEAN = math.sqrt(TPR * TNR)
    return G_MEAN
def matrix_verification(y_verification, y_pred):
    TP_count, TN_count, FP_count, FN_count = 0, 0, 0, 0
    for x, y in zip(y_verification, y_pred):
        if x == 1 and y == 1:
            TP_count = TP_count + 1
        elif x == 0 and y == 1:
            FP_count = FP_count + 1
        elif x == 0 and y == 0:
            TN_count = TN_count + 1
        elif x == 1 and y == 0:
            FN_count = FN_count + 1
    
    print("tn is:", TN_count)
    print("fp is:", FP_count)
    print("fn is:", FN_count)
    print("tp is:", TP_count)

def target1(clf, split):
    data = read_Data(r"C:\Users\zhipe\Desktop\target1.csv")
    data = data[data.TOTHLOS != -99]
    y = data.TOTHLOS
    #X = data.drop(['TOTHLOS','HXCHF','DIALYSIS','Groups','PRSEPIS'], axis=1)
    X = data.drop(['TOTHLOS'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=split, random_state=42, stratify=None)
    #clf = RandomForestClassifier(n_estimators=100)
    #clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=1e-3, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
    #clf = LogisticRegression(dual=False, random_state=10, max_iter=1000)
    #clf = DecisionTreeClassifier()
    #clf = AdaBoostClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # score = clf.score(X_test, y_test.ravel())
    # print(score)
    # print('Feature TOTHLOS')
    # print("Accuracy = %.2f" % (accuracy_score(y_test, y_pred)))
    # print("Recall =  %.2f" % recall_score(y_test, y_pred, average='micro'))
    # print("Precision =  %.2f" % precision_score(y_test, y_pred, average='micro'))
    # print("F1_score =  %.2f" % f1_score(y_test, y_pred, average='weighted'))
    print(confusion_matrix(y_test, y_pred))
    matrix = confusion_matrix(y_test, y_pred).tolist()
    print("G_mean score is %.2f" % g_Mean_class_0(matrix))
    print()
    

def target2(clf, split):
    data = read_Data(r"C:\Users\zhipe\Desktop\target2.csv")
    y = data.Mortality_1
    #X = data.drop(['Mortality_1','HXCHF','DIALYSIS','Groups','PRSEPIS'], axis=1)
    X = data.drop(['Mortality_1'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=split, random_state=42, stratify=None)
    #clf = RandomForestClassifier(n_estimators=100)
    #clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=1e-3, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
    #clf = LogisticRegression(dual=False, random_state=10, max_iter=1000)
    #clf = DecisionTreeClassifier()
    #clf = AdaBoostClassifier()
    clf.fit(X_train, y_train.ravel())
    y_pred = clf.predict(X_test)
    #score = clf.score(X_test, y_test.ravel())
    #print(score)
    # print('Feature Mortality_1')
    # print("Accuracy = %.2f" % (accuracy_score(y_test, y_pred)))
    # print("Recall =  %.2f" % recall_score(y_test, y_pred, average='micro'))
    # print("Precision =  %.2f" % precision_score(y_test, y_pred, average='micro'))
    # print("F1_score =  %.2f" % f1_score(y_test, y_pred, average='weighted'))
    print(confusion_matrix(y_test, y_pred))
    matrix = confusion_matrix(y_test, y_pred).tolist()
    print("G_mean score is %.2f" % g_Mean_class_0(matrix))
    print()
    #print(mean_squared_error(y_test, y_pred))


def target3(clf, split):
    data = read_Data(r"C:\Users\zhipe\Desktop\target3.csv")
    y = data.Reoperation_1
    #X = data.drop(['Reoperation_1','HXCHF','DIALYSIS','Groups','PRSEPIS'], axis=1)
    X = data.drop(['Reoperation_1'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=split, random_state=42, stratify=None)
    #clf = RandomForestClassifier(n_estimators=100)
    #clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=1e-3, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
    #clf = LogisticRegression(dual=False, random_state=10, max_iter=1000)
    #clf = DecisionTreeClassifier()
    #clf = AdaBoostClassifier()
    clf.fit(X_train, y_train.ravel())
    y_pred = clf.predict(X_test)
    #score = clf.score(X_test, y_test.ravel())
    #print(score)
    # print('Feature Reoperation_1')
    # print("Accuracy = %.2f" % (accuracy_score(y_test, y_pred)))
    # print("Recall =  %.2f" % recall_score(y_test, y_pred, average='micro'))
    # print("Precision =  %.2f" % precision_score(y_test, y_pred, average='micro'))
    # print("F1_score =  %.2f" % f1_score(y_test, y_pred, average='weighted'))
    print(confusion_matrix(y_test, y_pred))
    matrix = confusion_matrix(y_test, y_pred).tolist()
    print("G_mean score is %.2f" % g_Mean_class_0(matrix))
    print()
    #print(mean_squared_error(y_test, y_pred))

def target4(clf, split):
    data = read_Data(r"C:\Users\zhipe\Desktop\target4.csv")
    y = data.Readmission_1
    #X = data.drop(['Readmission_1','HXCHF','DIALYSIS','Groups','PRSEPIS'], axis=1)
    X = data.drop(['Readmission_1'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=split, random_state=42, stratify=None)
    #clf = RandomForestClassifier(n_estimators=100)
    #clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=1e-3, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
    #clf = LogisticRegression(dual=False, random_state=10, max_iter=1000)
    #clf = DecisionTreeClassifier()
    #clf = AdaBoostClassifier()
    clf.fit(X_train, y_train.ravel())
    y_pred = clf.predict(X_test)
    #score = clf.score(X_test, y_test.ravel())
    #print(score)
    # print('Feature Readmission_1')
    #print("Accuracy = %.2f" % (accuracy_score(y_test, y_pred)))
    print("Recall =  %.2f" % recall_score(y_test, y_pred))
    print("Precision =  %.2f" % precision_score(y_test, y_pred))
    print("F1_score =  %.2f" % f1_score(y_test, y_pred))
    matrix = confusion_matrix(y_test, y_pred).tolist()
    print("G_mean score is %.2f" % g_Mean_class_0(matrix))
    #print(mean_squared_error(y_test, y_pred))
    y_verification = [int(element) for element in y_test.tolist()]
    #print(type(y_pred.tolist()))
    matrix_verification(y_verification, y_pred)



    #new changes
    #print(confusion_matrix(y_test, y_pred))
    # matrix = confusion_matrix(y_test, y_pred).tolist()
    # print("G_mean score is %.2f" % g_Mean(matrix))
    # print()
   
   

if __name__ == "__main__":
    #clf = RandomForestClassifier(n_estimators=100)
    #clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=1e-3, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
    #clf = LogisticRegression(dual=False, random_state=10, max_iter=1000)
    #clf = DecisionTreeClassifier()
    clf = AdaBoostClassifier()
    # target1(clf, 0.2)#TO
    # target2(clf, 0.2)#MOTRALITY
    target3(clf, 0.2)#REOPERATION
    #target4(clf, 0.2)#READ
    


    #another experiment
    # a = [1,1,1,1,0,0,0,0,1,1]
    # b = [1,1,1,1,0,1,1,1,0,0]
    # u,w,h,k = confusion_matrix(a, b).ravel()
    # print(confusion_matrix(a, b))
    # print("Recall =  ", recall_score(a, b))
    # print("Precision = ", precision_score(a, b))
    # matrix_verification(a,b)
    # print()
    # print(u)
    # print(w)
    # print(h)
    # print(k)

   