"""
Test Module: TextCNN
Dataset: Kaggle-imdb-movie-reviews-dataset
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import seaborn as sns
from wordcloud import WordCloud
import time
import copy
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchtext import data
from torchtext.vocab import Vectors, GloVe
from sklearn.model_selection import train_test_split


def load_text_data(path, state="csv"):
    label = []
    if state == "FileFolder":
        text_data = []
        for dset in ["pos", "neg"]:
            path_dest = os.path.join(path, dset)
            path_list = os.listdir(path_dest)

            for fname in path_list:
                if fname.endswith(".txt"):
                    filename = os.path.join(path_dest, fname)
                    with open(filename) as f:
                        text_data.append(f.read())
                if dset == "pos":
                    label.append(1)
                else:
                    label.append(0)
            text_data = np.array(text_data)
            label = np.array(label)

    if state == "csv":
        dataset = pd.read_csv(path)
        # print('dataset columns: ', len(dataset.index), 'dataset indexs: ', len(dataset.columns))
        # dataset_df = pd.DataFrame(dataset, columns=dataset.columns, index=dataset.index)
        print(pd.value_counts(dataset.sentiment))
        X = np.array(dataset.iloc[:, 0].values)

        for sentiment in dataset.sentiment.values:
            if sentiment == "negative":
                label.append(0)
            if sentiment == "positive":
                label.append(1)
        label = np.array(label)
        # print(label)
        train_text, test_text, train_label, test_label = train_test_split(X, label, test_size=0.5, random_state=123)
    # return np.array(text_data), np.array(label)
    return train_text, test_text, train_label, test_label


def text_preprocess(text_data):
    text_pre = []
    for text in text_data:
        text = re.sub("<br /><br />", " ", text)
        text = text.lower()
        # 第二个参数表示是否保留减去的字符原本的占位空格 ""：不保留 " "：保留
        text = re.sub("\d+", "", text)
        text = text.translate(
            str.maketrans("", "", string.punctuation.replace("'", ""))
        )
        text = text.strip()
        text_pre.append(text)

    return np.array(text_pre)


def stop_stem_word(datalist, stop_words):
    datalist_pre = []
    for text in datalist:
        text_words = word_tokenize(text)
        text_words = [word for word in text_words if not word in stop_words]
        text_words = [word for word in text_words if len(re.findall("'", word)) == 0]
        datalist_pre.append(text_words)

    return np.array(datalist_pre, dtype=object)


def view_word_num_Freq(train_text, train_text_pre2, train_label, show=False):
    traindata = pd.DataFrame({"train_text": train_text, "train_word": train_text_pre2, "train_label": train_label})
    train_word_num = [len(text) for text in train_text_pre2]
    traindata["train_word_num"] = train_word_num
    plt.figure(figsize=(8, 5))
    _ = plt.hist(train_word_num, bins=100)
    plt.xlabel("Word Number")
    plt.ylabel("Freq")

    if show:
        plt.show()

    return traindata


if __name__ == '__main__':
    path = "../Dataset/IMDB-movie-reviews/IMDB Dataset.csv"
    train_text, test_text, train_label, test_label = load_text_data(path)

    train_text_pre = text_preprocess(train_text)
    test_text_pre = text_preprocess(test_text)

    stop_words = stopwords.words("english")
    stop_words = set(stop_words)
    train_text_pre2 = stop_stem_word(train_text_pre, stop_words)
    test_text_pre2 = stop_stem_word(test_text_pre, stop_words)
    print(train_text_pre[10000])
    print("="*10)
    print(train_text_pre2[10000])

    texts = [" ".join(words) for words in train_text_pre2]
    traindatasave = pd.DataFrame({"text":texts, "label":train_label})
    texts = [" ".join(words) for words in test_text_pre2]
    testdatasave = pd.DataFrame({"text":texts, "label":test_label})
    traindatasave.to_csv("../DataFrame/IMDB/imdb_train.csv", index=False)
    testdatasave.to_csv("../DataFrame/IMDB/imdb_test.csv", index=False)

    traindata = view_word_num_Freq(train_text, train_text_pre2, train_label, show=False)
    plt.figure(figsize=(16, 8))
    for ii in np.unique(train_label):
        text = np.array(traindata.train_word[traindata.train_label == ii])
        text = " ".join(np.concatenate(text))
        plt.subplot(1, 2, ii+1)
        wordcod = WordCloud(margin=5, width=1800, height=1000,
                            max_words=500, min_font_size=5,
                            background_color='white',
                            max_font_size=250)
        wordcod.generate_from_text(text)
        plt.imshow(wordcod)
        plt.axis("off")
        if ii == 1:
            plt.title("Positive")
        else:
            plt.title("Negative")
        plt.subplots_adjust(wspace=0.05)
    plt.show()


