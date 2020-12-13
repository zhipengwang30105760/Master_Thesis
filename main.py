import pandas as pd


def tpr_minus_fpr(matrix):
    tn = matrix[0][0]
    fp = matrix[0][1]
    fn = matrix[1][0]
    tp = matrix[1][1]

    try:
        score = (tp / (tp + fn)) - (fp / (tn + fp))
    except:
        score = 1
    return score

def tpr_times_tnr(matrix):
    tn = matrix[0][0]
    fp = matrix[0][1]
    fn = matrix[1][0]
    tp = matrix[1][1]

    try:
        score = (tp / (tp + fn)) * (tn / (tn + fp))
    except:
        score = 1
    return score

if __name__ == "__main__":

    confusion_matrix_list = [[[95989, 1288], [2203, 0]], [[97276, 1], [2203, 0]], [[97276, 1], [2203, 0]], [[96735, 542], [9, 2194]], [[97267, 5], [2203, 0]], [[27339, 69938], [0, 2203]], [[97209, 68], [74, 2129]], [[97254, 23], [2203, 0]], [[97266, 11], [2203, 0]], [[96707, 570], [2203, 0]], [[97044, 233], [2203, 0]], [[97234, 43], [2203, 0]], [[96833, 444], [2203, 0]], [[97277, 0], [2203, 0]], [[96906, 371], [2203, 0]], [[97162, 115], [2195, 8]], [[97180, 97], [2195, 8]], [[91841, 5436], [2140, 63]], [[91850, 5427], [2067, 136]], [[987, 96290], [0, 2203]], [[95027, 2250], [2197, 6]], [[86862, 10415], [1944, 259]], [[13635, 83642], [0, 2203]], [[91682, 5595], [2203, 0]], [[86412, 10865], [2195, 8]], [[97169, 108], [2203, 0]], [[97156, 121], [2203, 0]], [[97246, 31], [2203, 0]], [[91773, 5504], [2191, 12]], [[91608, 5669], [2191, 12]]]
    score_list = []
    for matrix in confusion_matrix_list:
        score = tpr_times_tnr(matrix)
        score_list.append(score)
    print(score_list)
    final = [score_list]

    binary_feature_collections =['protocol_type', 'land', 'urgent', 'hot', 'num_failed_logins',
                                  'logged_in', 'lnum_compromised',
                                  'lroot_shell', 'lsu_attempted', 'lnum_root', 'lnum_file_creations', 'lnum_shells',
                                  'lnum_access_files',
                                  'is_host_login', 'is_guest_login', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
                                  'srv_rerror_rate',
                                  'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_same_srv_rate',
                                  'dst_host_diff_srv_rate',
                                  'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                                  'dst_host_srv_serror_rate',
                                  'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
    df = pd.DataFrame(final, columns=binary_feature_collections)
    df.to_excel(r"/Users/zhipengwang/Desktop/output_result.xlsx", index=False)

