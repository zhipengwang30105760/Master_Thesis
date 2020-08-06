
import pandas as pd
import experiments_for_finding_best_featurres

feature_collections = ['Groups', 'SEX', 'AGE', 'BMI', 'SMOKE', 'DYSPNEA', 'FNSTATUS2', 'HXCOPD', 'ASCITES', 'HXCHF', 'HYPERMED', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDIS', 'TRANSFUS', 'PRSEPIS', 'ASACLAS', 'radial_all_yn', 'distal_all_yn', 'race_final', 'Emerg_yn', 'Diabetes_yn', 'Pre_staging', 'PATHO_staging']
noisy_features = ['AGE', 'SMOKE', 'FNSTATUS2', 'HXCOPD', 'DIALYSIS', 'TRANSFUS', 'radial_all_yn', 'PRSEPIS']
candidates_features = ['BMI', 'Groups', 'BLEEDIS', 'WTLOSS', 'HXCHF', 'ASACLAS', 'SEX', 'DYSPNEA', 'ASCITES', 'HYPERMED', 'DISCANCR', 'WNDINF', 'STEROID', 'distal_all_yn', 'race_final', 'Emerg_yn', 'Diabetes_yn', 'Pre_staging', 'PATHO_staging']
target = 'Mortality_1'
filename = r"/Users/zhipengwang/PycharmProjects/UNMC_Data_Analysis/data/undersampling_Mortality_1.csv"
data = pd.read_csv(filename)
y = data[target]
X = data.drop([target], axis=1)
#data normalization
scaler = experiments_for_finding_best_featurres.StandardScaler()
#do normalization for BMI and AGE
X.iloc[:, 2:4] = scaler.fit_transform(X.iloc[:, 2:4])
#choose the selected features
#X = data.loc[:,candidates_features]
#print(X.axes)
split1 = 0.3
split2 = 0.2
target_list = []
for features in feature_collections:

    sub_features = [features]
    X = data.loc[:, sub_features]
    try:
        r = experiments_for_finding_best_featurres.GENERATE_CONFUSION_MATRIX(X, y, split2, target)
    except:
        print("cannot get value")
        r= [[0, 0], [0, 0]]
    target_list.append(r)

print('===========================finished prediction==========================')

result = []
for a in target_list:
    #print(type(a[0][0]))
    score = (a[0][0] + a[1][1]) / (a[0][0] + a[1][1] + a[0][1] + a[1][0])
    result.append(score)
print('===========================finished calculation==========================')

thisdict = {}
for features, score in zip(feature_collections, result):
    thisdict[features] = score

sort_orders = sorted(thisdict.items(), key=lambda x: x[1], reverse=True)

print('===========================finished sorting==========================')

for k in sort_orders:
    print(k[0] + ": " + str(k[1]))

