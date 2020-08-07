import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

filename = r"/Users/zhipengwang/PycharmProjects/UNMC_Data_Analysis/data/undersampling_Mortality_1.csv"
data = pd.read_csv(filename)


def correlation_heatmap(train):
    correlations = train.corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=0.1, annot=True, cbar_kws={"shrink": 0.7})
    plt.show();


correlation_heatmap(data)