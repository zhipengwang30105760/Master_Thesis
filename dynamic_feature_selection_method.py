import pandas as pd
import math
import collections
from operator import itemgetter

def entropy_calculation(p, q):
    score = 100
    try:
        score = -sum([p[i] * math.log2(q[i]) for i in range(len(p))])
    except:
        print()
        #print('skip it')
    return score

def calculate_prerequiste_value(data, feature, target, visited):
    f_pos = 0
    tp = 0
    t_pos = 0
    tn = 0
    for i in range(len(data)):
        if i not in visited:
            if data[feature][i] == 1 and data[target][i] == 1:
                tp += 1
            if data[feature][i] == 0 and data[target][i] == 0:
                tn += 1
            if data[feature][i] == 1:
                f_pos += 1
            if data[target][i] == 1:
                t_pos += 1

    f_neg = len(data) - f_pos
    t_neg = len(data) - t_pos
    try:
        p1 = t_pos / len(data)
        p0 = t_neg / len(data)

        q1 = tp / f_pos
        q0 = tn / f_neg
    except:
        q1 = 0
        q0 = 0

    return p1, p0, q1, q0

def dynamic_selection(data, binary_feature_collections, target):
    #we need to remove the sample if the selected feature can also find the positive case
    num_of_positive_case = 0
    for i in range(len(data)):
        if data[target][i] == 1:
            num_of_positive_case += 1

    candidate_list = []
    visited = []
    #while they still pos case in target, do following
    while num_of_positive_case > 0:
        entropy_list = proposed_entropy_calculation(binary_feature_collections, data, target, visited)

        #sort the whole dict based on the entropy score, smaller means better
        entropy_list = sorted(entropy_list.items(), key=itemgetter(1))
        print(entropy_list)
        candidate_list.append(entropy_list[0][0])
        binary_feature_collections.remove(entropy_list[0][0])
        num_of_positive_case -= 1
        selected_feature = entropy_list[0][0]
        #drop the samples where feature value is 1 and target is also 1
        drop_determined_samples(selected_feature, visited)


    return candidate_list


def proposed_entropy_calculation(binary_feature_collections, data, target, visited):
    entropy_list = {}
    for feature in binary_feature_collections:
        # the first helper method
        p1, p0, q1, q0 = calculate_prerequiste_value(data, feature, target, visited)

        p = [p1, p0]
        q = [q1, q0]

        score = entropy_calculation(p, q)
        entropy_list[feature] = score
    return entropy_list


def drop_determined_samples(selected_feature, visited):
    for i in range(len(data)):
        if data[selected_feature][i] == 1 and data[target][i] == 1:
            visited.append(i)

if __name__ == "__main__":
    filename = r"/Users/zhipengwang/PycharmProjects/UNMC_Data_Analysis/data/original_DIED.csv"
    target="DIED"

    binary_feature_collections = ['FEMALE','CM_AIDS','CM_ALCOHOL','CM_ANEMDEF','CM_ARTH','CM_BLDLOSS','CM_CHF','CM_CHRNLUNG','CM_COAG','CM_DEPRESS','CM_DM','CM_DMCX','CM_DRUG','CM_HTN_C',
                           'CM_HYPOTHY','CM_LIVER','CM_LYMPH','CM_LYTES','CM_METS','CM_NEURO',
                           'CM_OBESE','CM_PARA','CM_PERIVASC','CM_PSYCH','CM_PULMCIRC','CM_RENLFAIL','CM_TUMOR','CM_ULCER','CM_VALVE','CM_WGHTLOSS']
    data = pd.read_csv(filename, encoding='ISO-8859-1')
    X = data
    entropy_list = dynamic_selection(data, binary_feature_collections, target)
    print(entropy_list)

