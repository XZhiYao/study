import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import torch.utils.data as Data
import matplotlib.pyplot as plt
import seaborn as sns
import hiddenlayer as hl
from torchviz import make_dot
import re


if __name__ == '__main__':
    spam = pd.read_csv("../Dataset/Spambase/spambase.csv")
    # print(spam.head())

    spam.columns = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our', 'word_freq_over',
              'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will',
              'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free', 'word_freq_business', 'word_freq_email',
              'word_freq_you', 'word_freq_credit', 'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money',
              'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs',
              'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology',
              'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct', 'word_freq_cs', 'word_freq_meeting',
              'word_freq_original', 'word_freq_project', 'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference',
              'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#',
              'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'label']
    # patten = 'word_'
    # re.match(patten, f)
    print(spam.head())
    print(pd.value_counts(spam.label))
    # print(len(spam))
    X = spam.iloc[:, 0:57].values
    y = spam.label.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

    scales = MinMaxScaler(feature_range=(0, 1))
    X_train_s = scales.fit_transform(X_train)
    X_test_s = scales.transform(X_test)

    colname = spam.columns.values[:-1]
    plt.figure(figsize=(20, 14))
    for ii in range(len(colname)):
        plt.subplot(7, 9, ii+1)
        sns.boxplot(x=y_train, y=X_train_s[:, ii])
        plt.title(colname[ii])

    plt.subplots_adjust(hspace=0.4)
    plt.show()