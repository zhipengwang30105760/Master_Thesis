from featexp import get_trend_stats, get_univariate_plots
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#using featexp to rank the features
from xverse.transformer import WOE

def generate_training_testing_set(dataframe, target_name):
    X = dataframe.drop(["Readmission_1"], axis=1)
    scaler = StandardScaler()
    X.iloc[:, 2:4] = scaler.fit_transform(X.iloc[:, 2:4])
    y = dataframe["Readmission_1"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    return (X_train, X_test, y_train, y_test)


def featexp_method(dataframe, target_name):
    X_train, X_test, y_train, y_test = generate_training_testing_set(dataframe, target_name)

    data_train = X_train.reset_index(drop=True)
    data_train['Readmission_1'] = y_train.reset_index(drop=True)
    data_test = X_test.reset_index(drop=True)
    data_test['Readmission_1'] = y_test.reset_index(drop=True)

    get_univariate_plots(data=data_train, target_col='Readmission_1', features_list=data_train.columns[21:22],
                         data_test=data_test)

    stats = get_trend_stats(data_train, target_col='Readmission_1', data_test=data_test)
    print(stats)

def woe_method(dataframes, target_name):
    X_train, X_test, y_train, y_test = generate_training_testing_set(dataframe, target_name)
    clf = WOE()
    clf.fit(X_train, y_train)
    print(clf.iv_df)



if __name__ == "__main__":
    # load data
    filename = r"/Users/zhipengwang/PycharmProjects/UNMC_Data_Analysis/data/oversampling_Readmission_1.csv"
    names = ['Groups', 'SEX', 'AGE', 'BMI', 'SMOKE', 'DYSPNEA', 'FNSTATUS2', 'HXCOPD', 'ASCITES', 'HXCHF', 'HYPERMED', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDIS', 'TRANSFUS', 'PRSEPIS', 'ASACLAS', 'radial_all_yn', 'distal_all_yn', 'race_final', 'Emerg_yn', 'Diabetes_yn', 'Pre_staging', 'PATHO_staging', 'Readmission_1']
    dataframe = read_csv(filename)
    array = dataframe.values
    target = 'Readmission_1'
    #featexp_method(dataframe, target)
    woe_method(dataframe, target)