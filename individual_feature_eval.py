import pandas as pd

m1k = []
m2k = []
mk = []
n1 = []
n2 = []
x1k = []
x2k = []
qk = []

def single_feature_evaluation(original_data, selected_column, target):

    data = original_data[[selected_column, target]]
    num_class1 = 0
    num_class2 = 0
    sum_class_1 = 0
    sum_class_2 = 0

    for i, j in data.iterrows():
        #j[0] is the feature value, j[1] is the target value
        if j[1] == 0.0:  #which is negative class, class1
            num_class1 += 1
            sum_class_1 += j[0]
        if j[1] == 1.0:  #which is positive class, class2
            num_class2 += 1
            sum_class_2 += j[0]
    mean_class1 = sum_class_1 / num_class1
    mean_class2 = sum_class_2 / num_class2

    mean_class1_and_class2 = (sum_class_1 + sum_class_2) / (num_class2 + num_class1)

    numerator = abs(mean_class1 - mean_class1_and_class2) + abs(mean_class2 - mean_class1_and_class2)
    m1k.append(mean_class1)
    m2k.append(mean_class2)
    mk.append(mean_class1_and_class2)
    n1.append(num_class1)
    n2.append(num_class2)

    sum_diff_class1 = 0
    sum_diff_class2 = 0

    for i, j in data.iterrows():
        # j[0] is the feature value, j[1] is the target value
        if j[1] == 0.0:  # which is negative class, class1
            sum_diff_class1 += abs(j[0] - mean_class1)
        if j[1] == 1.0:  # which is positive class, class2
            sum_diff_class2 += abs(j[0] - mean_class2)

    mean_diff_class1 = sum_diff_class1 / num_class1

    mean_diff_class2 = sum_diff_class2 / num_class2
    denominator = mean_diff_class1 + mean_diff_class2
    x1k.append(mean_diff_class1)
    x2k.append(mean_diff_class2)
    Qk = numerator / denominator
    qk.append(Qk)

    return Qk

def individual_feature_evaluation(original_data, feature_collections, target):
    dict = {}
    for feature in feature_collections:
        score = single_feature_evaluation(original_data, feature, target)
        dict[feature] = score;

    return dict

if __name__ == "__main__":
    filename = r"/Users/zhipengwang/PycharmProjects/UNMC_Data_Analysis/data/original_Mortality_1.csv"
    target = "Mortality_1"
    feature_collections = ['Groups', 'SEX', 'AGE', 'BMI', 'SMOKE', 'DYSPNEA', 'FNSTATUS2', 'HXCOPD', 'ASCITES', 'HXCHF',
                           'HYPERMED', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDIS', 'TRANSFUS',
                           'PRSEPIS', 'ASACLAS', 'radial_all_yn', 'distal_all_yn', 'race_final', 'Emerg_yn',
                           'Diabetes_yn', 'Pre_staging', 'PATHO_staging']


    original_data = pd.read_csv(filename)
    dict = individual_feature_evaluation(original_data, feature_collections, target)
    # print(dict)
    final = [m1k, m2k, mk, n1, n2, x1k, x2k, qk]
    df = pd.DataFrame(final, columns = feature_collections)
    df.to_excel(r"/Users/zhipengwang/Desktop/output_result.xlsx", index = False)