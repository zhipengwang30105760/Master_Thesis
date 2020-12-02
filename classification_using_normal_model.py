import pandas as pd
from itertools import permutations 
from sklearn.model_selection import train_test_split, cross_val_score
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
from TPR_and_TNR import TPR_and_TNR
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
    #model1 = svm.SVC(kernel='linear', C=1)
    model2 = KNeighborsClassifier(3)
    #model3 = RandomForestClassifier(n_estimators=100)
    model4 = LinearDiscriminantAnalysis()
    model5 = GaussianNB()

    #model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    #model3.fit(X_train, y_train)
    model4.fit(X_train, y_train)
    model5.fit(X_train, y_train)

    #y_pred1 = model1.predict(X_test)
    y_pred2 = model2.predict(X_test)
    #y_pred3 = model3.predict(X_test)
    y_pred4 = model4.predict(X_test)
    y_pred5 = model5.predict(X_test)

    #conf_mat1 = confusion_matrix(y_test, y_pred1)
    conf_mat2 = confusion_matrix(y_test, y_pred2)
    #conf_mat3 = confusion_matrix(y_test, y_pred3)
    conf_mat4 = confusion_matrix(y_test, y_pred4)
    conf_mat5 = confusion_matrix(y_test, y_pred5)

    #conf_mat_list.append(conf_mat1)
    conf_mat_list.append(conf_mat2)
    #conf_mat_list.append(conf_mat3)
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
def DIED_FEATURE_COLLECTIONS():
    feature_collections = ['CM_AIDS','CM_ALCOHOL','CM_ANEMDEF','CM_ARTH','CM_BLDLOSS','CM_CHF','CM_CHRNLUNG','CM_COAG','CM_DEPRESS','CM_DM'
        ,'CM_DMCX','CM_DRUG','CM_HTN_C','CM_HYPOTHY','CM_LIVER','CM_LYMPH','CM_LYTES','CM_METS','CM_NEURO','CM_OBESE','CM_PARA','CM_PERIVASC','CM_PSYCH'
        ,'CM_PULMCIRC','CM_RENLFAIL','CM_TUMOR','CM_ULCER','CM_VALVE','CM_WGHTLOSS','CM_RENLFAIL','CM_TUMOR']
    #candidate_features = ['CM_COAG', 'CM_CHF', 'CM_LYTES','CM_WGHTLOSS','CM_PULMCIRC','CM_RENLFAIL','CM_PERIVASC','CM_PARA','CM_TUMOR']
    #candidate_features = ['CM_CHF', 'CM_COAG', 'CM_LYTES', 'CM_PARA','CM_PERIVASC']
    #candidate_features = ['CM_CHF', 'CM_COAG', 'CM_LYTES', 'CM_WGHTLOSS', 'CM_PERIVASC']
    candidate_features = ['CM_CHRNLUNG', 'CM_ANEMDEF', 'CM_DM', 'CM_METS', 'CM_COAG']
    #candidate_features = ['CM_TUMOR','CM_PARA','CM_PERIVASC','CM_RENLFAIL']

def APPROACH_FEATURE_COLLECTIONS():
    # for ui
    # candidate_features = ['CM_LYTES','CM_METS','CM_WGHTLOSS','CM_ANEMDEF','CM_CHF','CM_BLDLOSS','CM_OBESE','CM_COAG','CM_PULMCIRC','CM_TUMOR']
    # for rfm
    # candidate_features = ['CM_ANEMDEF','CM_BLDLOSS','CM_CHF','CM_LYMPH','CM_LYTES','CM_METS','CM_TUMOR','CM_ULCER','CM_WGHTLOSS','CM_PARA']
    # for our method
    # candidate_features =['CM_OBESE','CM_HTN_C','CM_HYPOTHY','CM_DM','CM_ARTH','CM_DEPRESS','CM_DMCX','CM_CHRNLUNG','CM_PSYCH','CM_AIDS']
    # bottom 9
    # candidate_features = ['CM_ANEMDEF','CM_COAG','CM_PULMCIRC','CM_CHF','CM_METS','CM_BLDLOSS','CM_LYTES','CM_WGHTLOSS','CM_ULCER']

    candidate_features = ['CM_CHRNLUNG', 'CM_PSYCH', 'CM_AIDS', 'CM_DEPRESS', 'CM_DMCX']

def READMISSION_FEATURE_COLLECTIONS():
    #ui method
    #candidate_features = ['radial_all_yn','Emerg_yn','ASCITES','WTLOSS','HYPERMED','STEROID','TRANSFUS','distal_all_yn','race_final']
    #rfm method
    #candidate_features = ['HYPERMED','DIALYSIS','STEROID','radial_all_yn','race_final','Emerg_yn','distal_all_yn','DISCANCR','Diabetes_yn']
    #our method
    #candidate_features = ['Diabetes_yn','DISCANCR','SMOKE','WTLOSS','STEROID','BLEEDIS','HYPERMED','Emerg_yn','HXCOPD']
    candidate_features = ['distal_all_yn','radial_all_yn','race_final','ASCITES','SEX','TRANSFUS','DIALYSIS','HXCHF','WNDINF']

if __name__ == "__main__":
    #candidate_features = ['TRANSFUS','STEROID','radial_all_yn','SEX','race_final','Emerg_yn','WNDINF','HXCOPD','HXCHF']
    #candidate_features = ['WNDINF','STEROID','TRANSFUS','radial_all_yn','race_final','Emerg_yn','BLEEDIS','SMOKE','SEX']
    #candidate_features = ['WTLOSS','DISCANCR','STEROID','WNDINF','HXCHF','HXCOPD','Emerg_yn','ASCITES','TRANSFUS']
    #candidate_features = ['Emerg_yn','ASCITES','TRANSFUS','HXCHF','HXCOPD']
    candidate_features = ['race_final','radial_all_yn','distal_all_yn', 'SMOKE','Diabetes_yn','SEX']
    target = 'Readmission_1'
    filename = r"/Users/zhipengwang/PycharmProjects/UNMC_Data_Analysis/data/sample_Readmission_1.csv"
    data = pd.read_csv(filename)
    #remove the features with constant values
    #data, candidates_features = drop_constant_columns(data, candidate_features)
    y = data[target]
    #X = data.drop([target], axis=1)
    #X = data[candidate_features]
    #X = data[candidate_features[0: 5]]
    X = data[candidate_features[0: 3]]
    #start data normalization
    scaler = StandardScaler()
    #do normalization for BMI and AGE
    # X.iloc[:, 2:4] = scaler.fit_transform(X.iloc[:, 2:4])

    #tpr_and_tnr = TPR_and_TNR()
    a = TPR_and_TNR()

    # r = GENERATE_CONFUSION_MATRIX(X, y, split, target)
    # print(r)
    print('75/25')
    split = 0.25
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42, stratify=None)
    model2 = KNeighborsClassifier(3)
    model4 = LinearDiscriminantAnalysis()
    model5 = GaussianNB()

    model2.fit(X_train, y_train)
    model4.fit(X_train, y_train)
    model5.fit(X_train, y_train)

    y_pred2 = model2.predict(X_test)
    y_pred4 = model4.predict(X_test)
    y_pred5 = model5.predict(X_test)

    conf_mat2 = confusion_matrix(y_test, y_pred2)
    conf_mat4 = confusion_matrix(y_test, y_pred4)
    conf_mat5 = confusion_matrix(y_test, y_pred5)

    print(conf_mat2)
    print(conf_mat4)
    print(conf_mat5)
    print(a.tpr_minus_fpr(conf_mat2))
    print(a.tpr_minus_fpr(conf_mat4))
    print(a.tpr_minus_fpr(conf_mat5))


    print('80/20')
    split = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42, stratify=None)
    model2 = KNeighborsClassifier(3)
    model4 = LinearDiscriminantAnalysis()
    model5 = GaussianNB()

    model2.fit(X_train, y_train)
    model4.fit(X_train, y_train)
    model5.fit(X_train, y_train)

    y_pred2 = model2.predict(X_test)
    y_pred4 = model4.predict(X_test)
    y_pred5 = model5.predict(X_test)

    conf_mat2 = confusion_matrix(y_test, y_pred2)
    conf_mat4 = confusion_matrix(y_test, y_pred4)
    conf_mat5 = confusion_matrix(y_test, y_pred5)

    print(conf_mat2)
    print(conf_mat4)
    print(conf_mat5)
    print(a.tpr_minus_fpr(conf_mat2))
    print(a.tpr_minus_fpr(conf_mat4))
    print(a.tpr_minus_fpr(conf_mat5))

    print('70/30')
    split = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42, stratify=None)
    model2 = KNeighborsClassifier(3)
    model4 = LinearDiscriminantAnalysis()
    model5 = GaussianNB()

    model2.fit(X_train, y_train)
    model4.fit(X_train, y_train)
    model5.fit(X_train, y_train)

    y_pred2 = model2.predict(X_test)
    y_pred4 = model4.predict(X_test)
    y_pred5 = model5.predict(X_test)

    conf_mat2 = confusion_matrix(y_test, y_pred2)
    conf_mat4 = confusion_matrix(y_test, y_pred4)
    conf_mat5 = confusion_matrix(y_test, y_pred5)

    print(conf_mat2)
    print(conf_mat4)
    print(conf_mat5)
    print(a.tpr_minus_fpr(conf_mat2))
    print(a.tpr_minus_fpr(conf_mat4))
    print(a.tpr_minus_fpr(conf_mat5))

    #score = cross_val_score(svc, X, y, scoring = 'recall', cv=6)
    #print(score)





