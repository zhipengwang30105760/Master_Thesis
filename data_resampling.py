from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import pandas as pd

df_train = pd.read_csv(r"C:\Users\zhipe\Desktop\target4.csv")
count_class_0, count_class_1 = df_train.Readmission_1.value_counts()
#we seperate the class 0 and class 1 based on the target
df_class_0 = df_train[df_train['Readmission_1'] == 0]
df_class_1 = df_train[df_train['Readmission_1'] == 1]
#For undersampling
#df_class_0_under = df_class_0.sample(count_class_1)
#result = pd.concat([df_class_0_under, df_class_1], axis=0)
#For oversampling
df_class_1_over = df_class_1.sample(count_class_0, replace=True)
result = pd.concat([df_class_0, df_class_1_over], axis=0)

df = pd.DataFrame(result, columns = ['Groups', 'SEX', 'AGE', 'BMI', 'SMOKE', 'DYSPNEA', 'FNSTATUS2', 'HXCOPD', 'ASCITES', 'HXCHF', 'HYPERMED', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDIS', 'TRANSFUS', 'PRSEPIS', 'ASACLAS', 'radial_all_yn', 'distal_all_yn', 'race_final', 'Emerg_yn', 'Diabetes_yn', 'Pre_staging', 'PATHO_staging', 'Readmission_1'])
df.to_excel(r"C:\Users\zhipe\Desktop\oversampling_Readmission_1.xlsx", index = False)
#print(df_test_under.Readmission_1.value_counts())

