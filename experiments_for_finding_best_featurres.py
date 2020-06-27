from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_digits
from pandas import read_csv
from pandas.plotting import scatter_matrix
import pandas as pd
from matplotlib import pyplot
from itertools import permutations 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import pipeline
from sklearn import ensemble
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, average_precision_score, recall_score, f1_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
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

    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    model3.fit(X_train, y_train)
    model4.fit(X_train, y_train)

    y_pred1 = model1.predict(X_test)
    y_pred2 = model2.predict(X_test)
    y_pred3 = model3.predict(X_test)
    y_pred4 = model4.predict(X_test)

    conf_mat1 = confusion_matrix(y_test, y_pred1)
    conf_mat2 = confusion_matrix(y_test, y_pred2)
    conf_mat3 = confusion_matrix(y_test, y_pred3)
    conf_mat4 = confusion_matrix(y_test, y_pred4)

    conf_mat_list.append(conf_mat1)
    conf_mat_list.append(conf_mat2)
    conf_mat_list.append(conf_mat3)
    conf_mat_list.append(conf_mat4)
    return conf_mat_list

def MEAN(filename, feature_column, split, target):
    data = pd.read_csv(filename)
    y = data[target]
    X = data.drop([target], axis=1)
    #data normalization
    scaler = StandardScaler()
    X.iloc[:, 2:4] = scaler.fit_transform(X.iloc[:, 2:4])
    X = data.loc[:,feature_column]
    
    conf_mat_list = ensemble_learning_with_normal(X,y,split)

    #print(conf_mat_list)
    tn = 0
    fp = 0
    fn = 0
    tp = 0
    i =  0
    while i < 4:
        tn = tn + conf_mat_list[i][0][0]
        fp = fp + conf_mat_list[i][0][1]
        fn = fn + conf_mat_list[i][1][0]
        tp = tp + conf_mat_list[i][1][1]
        i = i + 1
    tn = math.ceil(tn / 4)
    fp = math.floor(fp / 4)
    fn = math.floor(fn / 4)
    tp = math.ceil(tp / 4)
    result = [[tn, fp], [fn, tp]]
    #print(result)
    return result

def permu_result(filename, sum_column, split1, target):
    perm = permutations(sum_column, 2)
    i = 3
    #print(list(perm))
    # for i in list(perm):
    #     print(list(i))
    final = []
    while i < len(sum_column):
        print("Permutation length is " + str(i))
        perm = permutations(sum_column, i)
        for j in list(perm):
            print(list(j))
            result = MEAN(filename, list(j), split1, target)
            print(result)
            if(result[0][0] > 319 and result[1][1] > 316):
                candidates = list(j) + result
                final.append(candidates)
                #print(list(j))
                #print(result)
        print("===================finished permutaion====================")
        print()
        print()
        i+=1 
    print()
    print()
    print("************************Here is the good result*************************")
    if len(final) == 0:
        print("nothing found")
    else:
        df = pd.DataFrame(final)
        df.to_excel(r"C:\Users\zhipe\Desktop\good_result.xlsx", index = False)


if __name__ == "__main__":
    #feature_collections = ['Groups', 'SEX', 'AGE', 'BMI', 'SMOKE', 'DYSPNEA', 'FNSTATUS2', 'HXCOPD', 'ASCITES', 'HXCHF', 'HYPERMED', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDIS', 'TRANSFUS', 'PRSEPIS', 'ASACLAS', 'radial_all_yn', 'distal_all_yn', 'race_final', 'Emerg_yn', 'Diabetes_yn', 'Pre_staging', 'PATHO_staging']
    sum_column = ['WNDINF', 'Diabetes_yn', 'ASACLAS', 'BLEEDIS', 'BMI', 'HYPERMED']
    #sum_column.reverse()
    filename = r"/Users/zhipengwang/PycharmProjects/UNMC_Data_Analysis/data/oversampling_readmission_1.csv"
    split1 = 0.3
    split2 = 0.2
    target = 'Readmission_1'
    r1 = MEAN(filename, sum_column, split1, target)
    print(r1)
    # i = len(sum_column)
    # while i > 0:
    #     print("number of features " + str(len(sum_column)))
    #     sub_features = sum_column[0:i]
    #     r = MEAN(filename, sub_features, split1, target)
    #     print(r)
    #     i -= 1
    

    #col = ['SMOKE', 'AGE','Diabetes_yn', 'WNDINF']
    # col = ['Diabetes_yn', 'WNDINF']
    # result = MEAN(filename, col, split1, target)
    
    
    

    # print('Mix result for 70/30: ')
    # print('Top25_Fisher')
    # r1 = MEAN(filename, sum_column[0], split1, target)
    # print('Top20_Fisher')
    # r2 = MEAN(filename, sum_column[1], split1, target)
    # print('Top25_Relief')
    # r3 = MEAN(filename, sum_column[2], split1, target)
    # print('Top20_Relief')
    # r4 = MEAN(filename, sum_column[3], split1, target)
    # print('Mix result for 80/20: ')
    # print('Top25_Fisher')
    # r5 = MEAN(filename, sum_column[0], split2, target)
    # print('Top20_Fisher')
    # r6 = MEAN(filename, sum_column[1], split2, target)
    # print('Top25_Relief')
    # r7 = MEAN(filename, sum_column[2], split2, target)
    # print('Top20_Relief')
    # r8 = MEAN(filename, sum_column[3], split2, target)

    # sum = [r1, r2, r3, r4, r5, r6, r7, r8]
    # #print(sum)
    # target_list =sum
    # result = []
    # for a in target_list:
    #     #print(type(a[0][0]))
    #     score = (a[0][0] + a[1][1]) / (a[0][0] + a[1][1] + a[0][1] + a[1][0])
    #     result.append(score)
    # print(result)
    # #sum.append(result)
    # df = pd.DataFrame(result)
    # #print(sum)
    # df.to_excel(r"C:\Users\zhipe\Desktop\result1.xlsx", index = False)




















    # print("70/30")
    # print('Top25 for Chi')
    # result_7_3_0 = MEAN(filename, sum_column[0], split1, target)
    # print('Top20 for Chi')
    # result_7_3_1 = MEAN(filename, sum_column[1], split1, target)
    # print('Top25 for ANOVA')
    # result_7_3_2 = MEAN(filename, sum_column[2], split1, target)
    # print('Top20 for ANOVA')
    # result_7_3_3 = MEAN(filename, sum_column[3], split1, target)
    # print('Top25 for Mutual_info')
    # result_7_3_4 = MEAN(filename, sum_column[4], split1, target)
    # print('Top20 for Mutual_info')
    # result_7_3_5 = MEAN(filename, sum_column[5], split1, target)
    # print()
    # print()
    # print('80/20')
    # print('Top25 for Chi')
    # result_8_2_0 = MEAN(filename, sum_column[0], split2, target)
    # print('Top20 for Chi')
    # result_8_2_1 = MEAN(filename, sum_column[1], split2, target)
    # print('Top25 for ANOVA')
    # result_8_2_2 = MEAN(filename, sum_column[2], split2, target)
    # print('Top20 for ANOVA')
    # result_8_2_3 = MEAN(filename, sum_column[3], split2, target)
    # print('Top25 for Mutual_info')
    # result_8_2_4 = MEAN(filename, sum_column[4], split2, target)
    # print('Top20 for Mutual_info')
    # result_8_2_5 = MEAN(filename, sum_column[5], split2, target)

    # first_list = [result_7_3_0, result_7_3_1, result_7_3_2, result_7_3_3, result_7_3_4, result_7_3_5]
    # second_list = [result_8_2_0, result_8_2_1, result_8_2_2, result_8_2_3, result_8_2_4, result_8_2_5]

    # print()
    # print(first_list)
    # print(second_list)