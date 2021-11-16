import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
fonts = FontProperties(fname="/Library/Fonts/华文细黑。ttf")
import re
import string
import copy
import time
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import jieba
from torchtext import data
from torchtext.vocab import Vectors


def chinese_pre(text_data):
    text_data = text_data.lower()
    text_data = re.sub("\d+", "", text_data)
    text_data = list(jieba.cut(text_data, cut_all=False))
    text_data = [word.strip() for word in text_data if word not in stop_word.text.values]
    text_data = " ".join(text_data)
    return text_data


if __name__ == '__main__':
    train_df = pd.read_csv("../Dataset/THUCNews/cnews.train.txt", sep="\t",
                           header=None, names=["label", "text"])
    val_df = pd.read_csv("../Dataset/THUCNews/cnews.val.txt", sep="\t",
                           header=None, names=["label", "text"])
    test_df = pd.read_csv("../Dataset/THUCNews/cnews.test.txt", sep="\t",
                         header=None, names=["label", "text"])
    stop_word = pd.read_csv("../Dataset/THUCNews/Chinese_Stopword_Dict.txt", encoding='gbk',
                         header=None, names=["text"])

    train_df["cutword"] = train_df.text.apply(chinese_pre)
    val_df["cutword"] = val_df.text.apply(chinese_pre)
    test_df["cutword"] = test_df.text.apply(chinese_pre)
    print(train_df.cutword.head())

