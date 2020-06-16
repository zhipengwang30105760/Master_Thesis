from pandas import read_csv
from numpy import set_printoptions
import numpy as np
from skfeature.function.similarity_based.lap_score import lap_score, feature_ranking
from skfeature.utility.construct_W import construct_W
from sklearn.feature_selection import SelectKBest
from scipy.sparse import *
from skfeature.function.similarity_based.reliefF import reliefF
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



def get_list_func(listName, reverse, Number):
    columns = ['Groups', 'SEX', 'AGE', 'BMI', 'SMOKE', 'DYSPNEA', 'FNSTATUS2', 'HXCOPD', 'ASCITES', 'HXCHF', 'HYPERMED', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDIS', 'TRANSFUS', 'PRSEPIS', 'ASACLAS', 'radial_all_yn', 'distal_all_yn', 'race_final', 'Emerg_yn', 'Diabetes_yn', 'Pre_staging', 'PATHO_staging']
    refer = {}
    for key, value in zip(columns, listName):
        add_value = value[0]
        refer[key] = add_value
    sort_orders = sorted(refer.items(), key=lambda x: x[1], reverse=True)
    result = []
    for i in sort_orders[0:Number]:
        result.append(i[0])
    return result

def PRINTLIST(list1, list2):
    # print('for us')
    feature_column_25_1 = get_list_func(list1, True, 25)
    #print(feature_column_25_1)
    feature_column_20_1 = get_list_func(list1, True, 20)
    #print(feature_column_20_1)
    #print('for rfm')
    feature_column_25_2 = get_list_func(list2, False, 25)
    #print(feature_column_25_2)
    feature_column_20_2 = get_list_func(list2, False, 20)
    #print(feature_column_20_2)
   # print('for fi')
    #feature_column_25_3 = get_list_func(list3, True, 25)
    #print(feature_column_25_3)
    #feature_column_20_3 = get_list_func(list3, True, 20)
    #print(feature_column_20_3)
    #summary = [feature_column_25_1, feature_column_20_1, feature_column_25_2, feature_column_20_2, feature_column_25_3, feature_column_20_3]
    summary=[feature_column_25_1, feature_column_20_1, feature_column_25_2, feature_column_20_2]
    print(summary)

def find_Common(list1, list2, list3):
    top10_first_feature = get_list_func(list1, True, 10)
    #print(top10_first_feature)
    top10_second_feature = get_list_func(list2, True, 10)
    #print(top10_second_feature)
    top10_third_feature = get_list_func(list3, True, 10)
    #print(top10_third_feature)
    final_result = []
    for i in top10_first_feature:
        final_result.append(i)
    for j in top10_second_feature:
        final_result.append(j)
    for k in top10_third_feature:
        final_result.append(k)
    final_result = set(final_result)
    #print(final_result)
    return final_result
def fisher_score(X, y):
    # Construct weight matrix W in a fisherScore way
    kwargs = {"neighbor_mode": "supervised", "fisher_score": True, 'y': y}
    W = construct_W(X, **kwargs)

    # build the diagonal D matrix from affinity matrix W
    D = np.array(W.sum(axis=1))
    L = W
    tmp = np.dot(np.transpose(D), X)
    D = diags(np.transpose(D), [0])
    Xt = np.transpose(X)
    t1 = np.transpose(np.dot(Xt, D.todense()))
    t2 = np.transpose(np.dot(Xt, L.todense()))
    # compute the numerator of Lr
    D_prime = np.sum(np.multiply(t1, X), 0) - np.multiply(tmp, tmp)/D.sum()
    # compute the denominator of Lr
    L_prime = np.sum(np.multiply(t2, X), 0) - np.multiply(tmp, tmp)/D.sum()
    # avoid the denominator of Lr to be 0
    D_prime[D_prime < 1e-12] = 10000
    lap_score = 1 - np.array(np.multiply(L_prime, 1/D_prime))[0, :]

    # compute fisher score from laplacian score, where fisher_score = 1/lap_score - 1
    score = 1.0/lap_score - 1
    fisher = [['Fisher_score']]
    revise = covertToList(score)
    #revise = word_tokenize(fit.scores_)
    #revise.append(fisher)
    return revise

def imp_reliefF(X, Y):
    relF = reliefF(X, Y)
    result = [['Relief_F']]
    revise = covertToList(relF)
    #revise.append(result)
    return revise

if __name__ == "__main__":    
    # load data
    filename = r"C:\Users\zhipe\Desktop\undersampling_Reoperation_1.csv"
    names = ['Groups', 'SEX', 'AGE', 'BMI', 'SMOKE', 'DYSPNEA', 'FNSTATUS2', 'HXCOPD', 'ASCITES', 'HXCHF', 'HYPERMED', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDIS', 'TRANSFUS', 'PRSEPIS', 'ASACLAS', 'radial_all_yn', 'distal_all_yn', 'race_final', 'Emerg_yn', 'Diabetes_yn', 'Pre_staging', 'PATHO_staging', 'Mortality_1']
    dataframe = read_csv(filename)
    array = dataframe.values
    for i in range(len(array)):
        for j in range(len(array[i])):
            value = float(array[i][j])
            array[i][j] = value
    #the last one is target
    X = array[:,0:27]
    Y = array[:,27]
    fisherlist = fisher_score(X,Y)
    relieflist = imp_reliefF(X, Y)
    
    #result = [fisherlist, relieflist]
    #print(list(fisherlist))
    #print(fisherlist)
    uslist = US(X,Y)
    PRINTLIST(fisherlist, relieflist)
    # #print(uslist)
    # #print(get_list_func(uslist, True, 25))
    #rfmlist = RFM(X,Y)
    # #print(rfmlist)
    #filist = FI(X,Y)
    # PRINTLIST(uslist, rfmlist, filist)
    #print(filist)
    #result = [uslist, rfmlist, filist]
    
    # chi2list = chi_Square(X,Y)
    # ANOVAlist = ANOVA(X,Y)
    # mutuallist = mutual_information(X,Y)
    # set1 = find_Common(chi2list, ANOVAlist, mutuallist)
    # set2 = find_Common(uslist, rfmlist, filist)
    # list(set1).append(list(set2))
    # print(list(set(set1)))
    #PRINTLIST(chi2list, ANOVAlist, mutuallist)
    #result = [chi2list, ANOVAlist, mutuallist]
    #print(len(relieflist))
    #df = pd.DataFrame(result, columns = ['Groups', 'SEX', 'AGE', 'BMI', 'SMOKE', 'DYSPNEA', 'FNSTATUS2', 'HXCOPD', 'ASCITES', 'HXCHF', 'HYPERMED', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDIS', 'TRANSFUS', 'PRSEPIS', 'ASACLAS', 'radial_all_yn', 'distal_all_yn', 'race_final', 'Emerg_yn', 'Diabetes_yn', 'Pre_staging', 'PATHO_staging'])
    #df.to_excel(r"C:\Users\zhipe\Desktop\result3.xlsx", index = False)
 
