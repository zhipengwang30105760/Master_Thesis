import pandas as pd
from itertools import permutations 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.model_selection import cross_val_predict
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def ensemble_learning_with_cross_validation(X,y):
    conf_mat_list = []
    model1 = svm.SVC(kernel='linear', C=1)
    model2 = KNeighborsClassifier(3)
    model3 = RandomForestClassifier(n_estimators=100)
    model4 = LinearDiscriminantAnalysis()

    y_pred1 = cross_val_predict(model1, X, y, cv=5)
    y_pred2 = cross_val_predict(model2, X, y, cv=5)
    y_pred3 = cross_val_predict(model3, X, y, cv=5)
    y_pred4 = cross_val_predict(model4, X, y, cv=5)

    conf_mat1 = confusion_matrix(y, y_pred1)
    conf_mat2 = confusion_matrix(y, y_pred2)
    conf_mat3 = confusion_matrix(y, y_pred3)
    conf_mat4 = confusion_matrix(y, y_pred4)

    conf_mat_list.append(conf_mat1)
    conf_mat_list.append(conf_mat2)
    conf_mat_list.append(conf_mat3)
    conf_mat_list.append(conf_mat4)

    return conf_mat_list

def ensemble_learning_with_normal(X,y,split):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=split, random_state=42, stratify=None)
    conf_mat_list = []
    model1 = svm.SVC(kernel='linear', C=1)
    model2 = KNeighborsClassifier(3)
    model3 = RandomForestClassifier(n_estimators=100)
    model4 = LinearDiscriminantAnalysis()
    model5 = GaussianNB()

    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    model3.fit(X_train, y_train)
    model4.fit(X_train, y_train)
    model5.fit(X_train, y_train)

    y_pred1 = model1.predict(X_test)
    y_pred2 = model2.predict(X_test)
    y_pred3 = model3.predict(X_test)
    y_pred4 = model4.predict(X_test)
    y_pred5 = model5.predict(X_test)

    conf_mat1 = confusion_matrix(y_test, y_pred1)
    conf_mat2 = confusion_matrix(y_test, y_pred2)
    conf_mat3 = confusion_matrix(y_test, y_pred3)
    conf_mat4 = confusion_matrix(y_test, y_pred4)
    conf_mat5 = confusion_matrix(y_test, y_pred5)

    conf_mat_list.append(conf_mat1)
    conf_mat_list.append(conf_mat2)
    conf_mat_list.append(conf_mat3)
    conf_mat_list.append(conf_mat4)
    conf_mat_list.append(conf_mat5)

    return conf_mat_list

def GENERATE_CONFUSION_MATRIX(X, y, split, target):

    conf_mat_list = ensemble_learning_with_normal(X,y,split)

    #print(conf_mat_list)
    tn = 0
    fp = 0
    fn = 0
    tp = 0
    i =  0
    length = len(conf_mat_list)
    while i < length:
        tn = tn + conf_mat_list[i][0][0]
        fp = fp + conf_mat_list[i][0][1]
        fn = fn + conf_mat_list[i][1][0]
        tp = tp + conf_mat_list[i][1][1]
        i = i + 1
    tn = math.ceil(tn / length)
    fp = math.floor(fp / length)
    fn = math.floor(fn / length)
    tp = math.ceil(tp / length)
    result = [[tn, fp], [fn, tp]]
    return result

def drop_constant_columns(dataframe, candidate_features):
    candidates_features = candidate_features.copy()
    result = dataframe.copy()
    for column in dataframe.columns:
        if len(dataframe[column].unique()) == 1:
            result = result.drop(column,axis=1)
            if column in feature_collections:
                feature_collections.remove(column)
    return result, candidates_features


if __name__ == "__main__":
    feature_collections = ['Groups', 'SEX', 'AGE', 'BMI', 'SMOKE', 'DYSPNEA', 'FNSTATUS2', 'HXCOPD', 'ASCITES', 'HXCHF', 'HYPERMED', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDIS', 'TRANSFUS', 'PRSEPIS', 'ASACLAS', 'radial_all_yn', 'distal_all_yn', 'race_final', 'Emerg_yn', 'Diabetes_yn', 'Pre_staging', 'PATHO_staging']
    # noisy_features = ['AGE', 'SMOKE', 'FNSTATUS2', 'HXCOPD', 'DIALYSIS', 'TRANSFUS', 'radial_all_yn', 'PRSEPIS']

    candidates_features = ['Groups','SEX','SMOKE','radial_all_yn','WNDINF']
    target = 'Readmission_1'
    filename = r"/Users/zhipengwang/PycharmProjects/UNMC_Data_Analysis/data/original_Readmission_1.csv"
    data = pd.read_csv(filename)
    #remove the features with constant values
    data, candidates_features = drop_constant_columns(data, candidates_features)
    y = data[target]
    #X = data.drop([target], axis=1)
    X = data[candidates_features]
    #start data normalization
    scaler = StandardScaler()
    #do normalization for BMI and AGE
    X.iloc[:, 2:4] = scaler.fit_transform(X.iloc[:, 2:4])
    split = 0.2

    r = GENERATE_CONFUSION_MATRIX(X, y, split, target)
    print(r)




