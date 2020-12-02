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
import TPR_and_TNR

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
    rfe = RFE(model, 9)
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
def get_list_func(listName, reverse):
    # columns = ['CM_AIDS','CM_ALCOHOL','CM_ANEMDEF','CM_ARTH','CM_BLDLOSS','CM_CHF','CM_CHRNLUNG','CM_COAG','CM_DEPRESS','CM_DM'
    #     ,'CM_DMCX','CM_DRUG','CM_HTN_C','CM_HYPOTHY','CM_LIVER','CM_LYMPH','CM_LYTES','CM_METS','CM_NEURO','CM_OBESE','CM_PARA','CM_PERIVASC','CM_PSYCH'
    #     ,'CM_PULMCIRC','CM_RENLFAIL','CM_TUMOR','CM_ULCER','CM_VALVE','CM_WGHTLOSS','CM_RENLFAIL','CM_TUMOR']
    columns =['SEX', 'SMOKE', 'HXCOPD', 'ASCITES', 'HXCHF',
                                  'HYPERMED', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDIS',
                                  'TRANSFUS', 'radial_all_yn', 'distal_all_yn', 'race_final', 'Emerg_yn',
                                  'Diabetes_yn']

    refer = {}
    for key, value in zip(columns, listName):
        add_value = value[0]
        refer[key] = add_value
    sort_orders = sorted(refer.items(), key=lambda x: x[1], reverse=reverse)
    result = []
    for i in sort_orders[0:len(columns)]:
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
        output_sorted_result = get_list_func(i, True)
        print(output_sorted_result)

def write_output(output_content):
    df = pd.DataFrame(output_content,
                      columns=['CM_AIDS','CM_ALCOHOL','CM_ANEMDEF','CM_ARTH','CM_BLDLOSS','CM_CHF','CM_CHRNLUNG','CM_COAG','CM_DEPRESS','CM_DM'
        ,'CM_DMCX','CM_DRUG','CM_HTN_C','CM_HYPOTHY','CM_LIVER','CM_LYMPH','CM_LYTES','CM_METS','CM_NEURO','CM_OBESE','CM_PARA','CM_PERIVASC','CM_PSYCH'
        ,'CM_PULMCIRC','CM_RENLFAIL','CM_TUMOR','CM_ULCER','CM_VALVE','CM_WGHTLOSS','CM_RENLFAIL','CM_TUMOR'])
    df.to_excel(r"/Users/zhipengwang/PycharmProjects/UNMC_Data_Analysis/data/temporary_output.xlsx", index=False)

if __name__ == "__main__":    
    # load data
    filename = r"/Users/zhipengwang/PycharmProjects/UNMC_Data_Analysis/data/sample_Reoperation_1.csv"
    # feature_collections = ['CM_AIDS','CM_ALCOHOL','CM_ANEMDEF','CM_ARTH','CM_BLDLOSS','CM_CHF','CM_CHRNLUNG','CM_COAG','CM_DEPRESS','CM_DM'
    #     ,'CM_DMCX','CM_DRUG','CM_HTN_C','CM_HYPOTHY','CM_LIVER','CM_LYMPH','CM_LYTES','CM_METS','CM_NEURO','CM_OBESE','CM_PARA','CM_PERIVASC','CM_PSYCH'
    #     ,'CM_PULMCIRC','CM_RENLFAIL','CM_TUMOR','CM_ULCER','CM_VALVE','CM_WGHTLOSS','CM_RENLFAIL','CM_TUMOR']
    # candidates_features = [x for x in feature_collections if x not in noisy_features]
    binary_feature_collections = ['SEX', 'SMOKE', 'HXCOPD', 'ASCITES', 'HXCHF',
                                  'HYPERMED', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDIS',
                                  'TRANSFUS', 'radial_all_yn', 'distal_all_yn', 'race_final', 'Emerg_yn',
                                  'Diabetes_yn']

    dataframe = read_csv(filename)
    target = 'Reoperation_1'
    Y_dataframe = dataframe[target]
    X_dataframe = dataframe.drop([target], axis=1)
    # data normalization
    # scaler = StandardScaler()
    # X_dataframe.iloc[:, 2:4] = scaler.fit_transform(X_dataframe.iloc[:, 2:4])
    # X_dataframe = dataframe.loc[:, candidates_features]

    X = X_dataframe.values
    Y = Y_dataframe.values

    # uslist = US(X,Y)
    # print(get_list_func(uslist, True))
    rfmlist = RFM(X,Y)
    print(get_list_func(rfmlist, False))
    #filist = FI(X,Y)
    # PRINTLIST(uslist, rfmlist, filist)
    #print(get_list_func(filist, True))
    # lasso(X, Y)
