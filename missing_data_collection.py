import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.impute import SimpleImputer
from pandas import ExcelFile,ExcelWriter
from openpyxl.workbook import Workbook
def Impute_For_Missing_Values():
    filename = r"C:\Users\zhipe\Desktop\Proctectomy dataset  1 original.xlsx"
    X = excel_Reader(filename)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    cool = np.array(imp.fit_transform(X)).tolist()
    result = [[round(a), round(b,2), round(c), round(d)] for a, b, c, d in cool]
    #print(result[334])
    return result
   
def excel_Reader(filename):
    df = pd.read_excel(filename)
    PATHO_staging = list(df['PATHO_staging'])
    Pre_staging = list(df['Pre_staging'])
    BMI = list(df['BMI'])
    AGE = list(df['AGE'])
    newList = []
    for age, bmi, patho, pre in zip(AGE, BMI, PATHO_staging, Pre_staging):
        newList.append([age, bmi, patho, pre])
    return newList
    #print(newList)
def insert_Back(result):
    # BMI =[]
    # PATHO_staging = []
    # Pre_staging = []
    # for a, b, c in result:
    #     BMI.append(a)
    #     PATHO_staging.append(b)
    #     Pre_staging.append(c)
    # print(BMI)
    # print(PATHO_staging)
    # print(Pre_staging)
    df = pd.DataFrame(result, columns = ['AGE', 'BMI', 'PATHO_staging', 'Pre_staging'])
    df.to_excel(r"C:\Users\zhipe\Desktop\output.xlsx", index = False)

if __name__ =="__main__":
    result = Impute_For_Missing_Values()
    insert_Back(result)