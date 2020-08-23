import itertools

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import math
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import numpy as np

def drop_constant_columns(dataframe, candidate_features):
    """
    Drops constant value columns of pandas dataframe.
    """
    candidates_features = candidate_features.copy()
    result = dataframe.copy()
    for column in dataframe.columns:
        if len(dataframe[column].unique()) == 1:
            result = result.drop(column,axis=1)
            if column in feature_collections:
                feature_collections.remove(column)
    return result, candidates_features

def generate_classifier(layer, activation, solver, alpha, learning_rate):
    classifier = MLPClassifier(hidden_layer_sizes=layer, max_iter=500, activation=activation, solver=solver,
                               random_state=1, alpha=alpha, learning_rate=learning_rate)
    return classifier

def hba(matrix):
    tn = matrix[0][0]
    fp = matrix[0][1]
    fn = matrix[1][0]
    tp = matrix[1][1]

    score = (2*tp*tn) / (2*tp*tn + fp*tp + tn*fn)
    return score

def gba(matrix):
    tn = matrix[0][0]

    fp = matrix[0][1]
    fn = matrix[1][0]
    tp = matrix[1][1]

    sum = (tp / (tp + fn)) * (tn / (tn + fp))
    score = math.sqrt(sum)
    return score

def custom_mlp(parameter_space):
    hidden_layer_size = parameter_space['hidden_layer_sizes']
    activations = parameter_space['activation']
    solvers = parameter_space['solver']
    alphas = parameter_space['alpha']
    learning_rates = parameter_space['learning_rate']
    all_combination=[]
    for layer in hidden_layer_size:
        for activation in activations:
            for solver in solvers:
                for alpha in alphas:
                    for learning_rate in learning_rates:
                        all_combination.append([layer, activation, solver, alpha, learning_rate])

    classifiers = []
    for comb in all_combination:
        classifier = generate_classifier(comb[0], comb[1], comb[2], comb[3], comb[4])
        classifiers.append(classifier)
    return classifiers


def configure_best_classifier(mlp):
    parameter_space = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,),(150, 50, 100)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=5)
    clf.fit(X_train, y_train)
    # Best paramete set
    print('Best parameters found:\n', clf.best_params_)

    # All results
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    y_true, y_pred = y_test, clf.predict(X_test)


    print('Results on the test set:')
    print(classification_report(y_true, y_pred))



    print()
    print(y_pred)

def using_keras():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3)
    ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer='adam')
    model.summary()



if __name__ == "__main__":
    filename = r"/Users/zhipengwang/PycharmProjects/UNMC_Data_Analysis/data/original_Mortality_1.csv"
    target="Mortality_1"
    feature_collections = ['Groups', 'SEX', 'AGE', 'BMI', 'SMOKE', 'DYSPNEA', 'FNSTATUS2', 'HXCOPD', 'ASCITES', 'HXCHF',
                           'HYPERMED', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID', 'WTLOSS', 'BLEEDIS', 'TRANSFUS',
                           'PRSEPIS', 'ASACLAS', 'radial_all_yn', 'distal_all_yn', 'race_final', 'Emerg_yn',
                           'Diabetes_yn', 'Pre_staging', 'PATHO_staging']
    data = pd.read_csv(filename)
    data, candidates_features = drop_constant_columns(data, feature_collections)
    y = data[target]
    X = data.drop([target], axis=1)
    # start data normalization
    scaler = StandardScaler()
    # do normalization for BMI and AGE
    X.iloc[:, 2:4] = scaler.fit_transform(X.iloc[:, 2:4])

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)
    #default mlp classifier
    #load classifier
    # Readmission MLPClassifier(hidden_layer_sizes=(50, 50, 50), max_iter=500, random_state=1)
    # Reoperation MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 100, 50), learning_rate='adaptive', max_iter=500, random_state=1)
    # Morality MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 50, 50), max_iter=500, random_state=1, solver='sgd')

    mlp = generate_classifier((50, 50, 50), 'tanh', 'sgd', 0.5, 'adaptive')



    #configure_best_classifier(mlp)
    for feature in feature_collections:
        sub_features = [feature]
        X = data.loc[:, sub_features]

        # start data normalization
        scaler = StandardScaler()
        # do normalization for BMI and AGE
        # if feature == 'BMI' or feature == 'AGE':
        #     X= scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)


        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        cm = confusion_matrix(y_pred, y_test)
        print('Feature name is: ' + feature)
        print(cm)

    #
    #custom classifier
    # parameter_space = {
    #     'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,), (150, 50, 100)],
    #     'activation': ['tanh', 'relu'],
    #     'solver': ['sgd', 'adam'],
    #     'alpha': [0.0001, 0.05],
    #     'learning_rate': ['constant', 'adaptive'],
    # }
    #
    # classifiers = custom_mlp(parameter_space)
    # print('start')
    # for classifier in classifiers:
    #     classifier.fit(X_train, y_train)
    #     y_pred = classifier.predict(X_test)
    #     cm = confusion_matrix(y_pred, y_test)
    #     if(cm[1][1] == 0):
    #         print(classifier)
    #         print(cm)














