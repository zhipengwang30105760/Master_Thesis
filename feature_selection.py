from pandas import read_csv
from numpy import set_printoptions
import numpy as np
from scipy.sparse import *
from skfeature.utility.construct_W import construct_W
#from skfeature.function.similarity_based.lap_score import lap_score, feature_ranking
from skfeature.utility.construct_W import construct_W
from sklearn.feature_selection import SelectKBest
from skfeature.function.similarity_based.reliefF import reliefF
from sklearn.feature_selection import f_classif, SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2, f_regression
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectPercentile, VarianceThreshold
from sklearn.model_selection import train_test_split

def covertToList(npArray):
    alist = []
    for a in npArray:
        sublist =[]
        sublist.append(a)
        alist.append(sublist)
    return alist
def US(X, Y):
    test = SelectKBest(score_func=f_classif, k=4)
    fit = test.fit(X, Y.ravel())
    # summarize scores
    set_printoptions(precision=4)
    #print(fit.scores_)
    covertToList(fit.scores_)
    result = [['Univariate_Selection']]
    revise = covertToList(fit.scores_)
    #revise = word_tokenize(fit.scores_)
    result.append(revise)
    return revise
def RFM(X,Y):
    model = LogisticRegression(solver='lbfgs')
    #fetch the most significant three features
    rfe = RFE(model, 3)         
    fit = rfe.fit(X, Y)
    # print("Num Features: %d" % fit.n_features_)
    # print("Selected Features: %s" % fit.support_)
    # print("Feature Ranking: %s" % fit.ranking_)
    result = [['Recursive Feature Elimination']]
    revise = covertToList(fit.ranking_)
    #revise = word_tokenize(fit.ranking_)
    result.append(revise)
    return revise

def PCA1(X,Y):
    pca = PCA(n_components=3)
    fit = pca.fit(X)
    print("Explained Variance: %s" % fit.explained_variance_ratio_)
    print(fit.components_)
    

def FI(X,Y):
    model = ExtraTreesClassifier(n_estimators=10)
    model.fit(X, Y)
    #print(model.feature_importances_)

    result = [['Feature_Importance']]
    revise = covertToList(model.feature_importances_)
    # revise = word_tokenize(model.feature_importances_)
    result.append(revise)
    return revise

def chi_Square(X, Y):
    # for col in df.columns:
    #     le = LabelEncoder()
    #     df[col] = le.fit_transform(df[col])
    chi2_result, pval = chi2(X, Y)
    #print(chi2_result)
    result = [['Chi_Sqaure']]
    revise = covertToList(chi2_result)
    #revise = word_tokenize(fit.scores_)
    result.append(revise)
    return revise

def ANOVA(X, Y):
    chi2_result, pval = f_classif(X, Y)
    np.round(chi2_result)
    #print(chi2_result)
    result = [['ANOVA']]
    revise = covertToList(chi2_result)
    #revise = word_tokenize(fit.scores_)
    result.append(revise)
    return revise

def mutual_information(X,Y):
    mutual = mutual_info_classif(X, Y)
    #print(result)
    result = [['Mutual_Information']]
    revise = covertToList(mutual)
    #revise = word_tokenize(fit.scores_)
    result.append(revise)
    return revise

def constant_Remove(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    feature_selector = VarianceThreshold(threshold = 0)
    feature_selector.fit(X_train)
    print(feature_selector.get_support())

def lasso(X,Y):
    sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))
    sel_.fit(X,Y)
    result = sel_.get_support()
    print(result)
    return result

#format the output function
def get_list_func(listName, reverse, Number):
    columns = ['Groups', 'SEX', 'BMI', 'DYSPNEA', 'ASCITES', 'HXCHF', 'HYPERMED',
       'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDIS', 'ASACLAS',
       'distal_all_yn', 'race_final', 'Emerg_yn', 'Diabetes_yn', 'Pre_staging',
       'PATHO_staging']
    refer = {}
    for key, value in zip(columns, listName):
        add_value = value[0]
        refer[key] = add_value
    sort_orders = sorted(refer.items(), key=lambda x: x[1], reverse=reverse)
    result = []
    for i in sort_orders[0:Number]:
        result.append(i[0])
    return result

def imp_reliefF(X, Y):
    relF = reliefF(X, Y)
    result = [['Relief_F']]
    revise = covertToList(relF)
    #revise.append(result)
    return revise
#sort all the features based on the ranking
def output_list(list_summary):
    for i in list_summary:
        output_sorted_result = get_list_func(i, True, 27)
        print(output_sorted_result)

def write_output(output_content):
    df = pd.DataFrame(output_content,
                      columns=['Groups', 'SEX', 'AGE', 'BMI', 'SMOKE', 'DYSPNEA', 'FNSTATUS2', 'HXCOPD', 'ASCITES',
                               'HXCHF', 'HYPERMED', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDIS',
                               'TRANSFUS', 'PRSEPIS', 'ASACLAS', 'radial_all_yn', 'distal_all_yn', 'race_final',
                               'Emerg_yn', 'Diabetes_yn', 'Pre_staging', 'PATHO_staging'])
    df.to_excel(r"/Users/zhipengwang/PycharmProjects/UNMC_Data_Analysis/data/temporary_output.xlsx", index=False)

if __name__ == "__main__":    
    # load data
    filename = r"/Users/zhipengwang/PycharmProjects/UNMC_Data_Analysis/data/oversampling_Readmission_1.csv"
    feature_collections = ['Groups', 'SEX', 'AGE', 'BMI', 'SMOKE', 'DYSPNEA', 'FNSTATUS2', 'HXCOPD', 'ASCITES', 'HXCHF', 'HYPERMED', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDIS', 'TRANSFUS', 'PRSEPIS', 'ASACLAS', 'radial_all_yn', 'distal_all_yn', 'race_final', 'Emerg_yn', 'Diabetes_yn', 'Pre_staging', 'PATHO_staging']
    noisy_features =['AGE', 'SMOKE', 'FNSTATUS2', 'HXCOPD', 'DIALYSIS', 'TRANSFUS', 'radial_all_yn', 'PRSEPIS']
    candidates_features = [x for x in feature_collections if x not in noisy_features]

    dataframe = read_csv(filename)
    target = 'Readmission_1'
    Y_dataframe = dataframe[target]
    X_dataframe = dataframe.drop([target], axis=1)
    # data normalization
    scaler = StandardScaler()
    X_dataframe.iloc[:, 2:4] = scaler.fit_transform(X_dataframe.iloc[:, 2:4])
    X_dataframe = dataframe.loc[:, candidates_features]

    X = X_dataframe.values
    Y = Y_dataframe.values


    #print(feature_collections)
    # lalist = lasso(X,Y)
    # for i in range(len(lalist)):
    #     if lalist[i] == False:
    #         print('index is ' + str(i))
    #         print('feature name is ' + feature_collections[i])
    #         print(lalist[i])

    mutuallist = mutual_information(X, Y)
    sorted_result = get_list_func(mutuallist, True, 19)
    print(sorted_result)
    
    # fisherlist = fisher_score(X,Y)
    # relieflist = imp_reliefF(X, Y)
    # filist = FI(X,Y)
    #
    # ANOVAlist = ANOVA(X,Y)
    # mutuallist = mutual_information(X,Y)
    # list_collector = [fisherlist, relieflist, filist, ANOVAlist, mutuallist]
    # for l in list_collector:
    #     print(l)
    # output_list(list_collector)



    #result = [fisherlist, relieflist]
    #print(list(fisherlist))
    #print(fisherlist)
    # uslist = US(X,Y)
    # PRINTLIST(fisherlist, relieflist)
    # #print(uslist)
    # #print(get_list_func(uslist, True, 25))
    #rfmlist = RFM(X,Y)
    # #print(rfmlist)
    #filist = FI(X,Y)
    # PRINTLIST(uslist, rfmlist, filist)
    #print(filist)
    #result = [uslist, rfmlist, filist]
 
