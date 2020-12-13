import pandas as pd
import math

m1k = []
m2k = []
mk = []
n1 = []
n2 = []
x1k = []
x2k = []
qk = []
score_list = []

def Qk(original_data, selected_column, target):

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

def projection_on_one_dimension(original_data, feature_collections, target):
    dict = {}
    for feature in feature_collections:
        score = Qk(original_data, feature, target)
        dict[feature] = score;

    return dict

def PCA(original_data, feature, target):
    mean_x = 0
    mean_y = 0
    numerator = 0
    denominator = 0
    deno_1 = 0
    deno_2 = 0

    for i in range(len(original_data)):
        mean_x += original_data[feature][i]
        mean_y += original_data[target][i]

    mean_x /= len(original_data)
    mean_y /= len(original_data)

    for i in range(len(original_data)):
        numerator += (original_data[feature][i] - mean_x) * (original_data[target][i] - mean_y)
        deno_1 += pow((original_data[feature][i] - mean_x), 2)
        deno_2 += pow((original_data[target][i] - mean_y), 2)

    denominator = math.sqrt(deno_1 * deno_2)
    return numerator / denominator


def pearson_corrleation_analysis(original_data, feature_collections, target):
    dict = {}
    for feature in feature_collections:
        score = PCA(original_data, feature, target)
        dict[feature] = score;
        score_list.append(score)

    return dict

if __name__ == "__main__":
    filename = r"/Users/zhipengwang/PycharmProjects/UNMC_Data_Analysis/data/kddcup99.csv"
    target = "label"
    feature_collections = ['protocol_type', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
                                  'logged_in', 'lnum_compromised',
                                  'lroot_shell', 'lsu_attempted', 'lnum_root', 'lnum_file_creations', 'lnum_shells',
                                  'lnum_access_files', 'lnum_outbound_cmds',
                                  'is_host_login', 'is_guest_login', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
                                  'srv_rerror_rate',
                                  'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_same_srv_rate',
                                  'dst_host_diff_srv_rate',
                                  'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                                  'dst_host_srv_serror_rate',
                                  'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
    feature_collections.remove('wrong_fragment')
    feature_collections.remove('lnum_outbound_cmds')

    original_data = pd.read_csv(filename)
    dict = projection_on_one_dimension(original_data, feature_collections, target)
    print(dict)
    # # final = [m1k, m2k, mk, n1, n2, x1k, x2k, qk]
    final = [dict]
    df = pd.DataFrame(final, columns = feature_collections)
    df.to_excel(r"/Users/zhipengwang/Desktop/output_result.xlsx", index = False)