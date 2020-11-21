
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

if __name__ == "__main__":

    confusion_matrix_list = [[[16827, 195],[560, 28]], [[16738, 284],[503, 85]], [[15050, 1972],[308, 280]], [[13109, 512],[428, 39]], [[13398, 223],[399, 68]], [[12067, 1554],[250, 217]],
                             [[20035, 383],[644, 70]], [[20082, 336],[612, 102]], [[17915, 2503],[373, 341]]]

    for matrix in confusion_matrix_list:
        score = tpr_minus_fpr(matrix)
        print(score)


