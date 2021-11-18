import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
fonts = FontProperties(fname=r"/System/Library/Fonts/Songti.ttc")
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
from torchtext.legacy import data
from torchtext.vocab import Vectors


class LSTMNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embeds = self.embedding(x)
        r_out, (h_n, h_c) = self.lstm(embeds, None)
        output = self.fc1(r_out[:, -1, :])
        return output


def chinese_pre(text_data):
    stop_word = pd.read_csv("../Dataset/THUCNews/Chinese_Stopword_Dict.txt", encoding='gbk',
                            header=None, names=["text"])

    text_data = text_data.lower()
    text_data = re.sub("\d+", "", text_data)
    text_data = list(jieba.cut(text_data, cut_all=False))
    text_data = [word.strip() for word in text_data if word not in stop_word.text.values]
    text_data = " ".join(text_data)
    return text_data


def prepare_dataset():
    train_df = pd.read_csv("../Dataset/THUCNews/cnews.train.txt", sep="\t",
                           header=None, names=["label", "text"])
    val_df = pd.read_csv("../Dataset/THUCNews/cnews.val.txt", sep="\t",
                         header=None, names=["label", "text"])
    test_df = pd.read_csv("../Dataset/THUCNews/cnews.test.txt", sep="\t",
                          header=None, names=["label", "text"])

    train_df["cutword"] = train_df.text.apply(chinese_pre)
    val_df["cutword"] = val_df.text.apply(chinese_pre)
    test_df["cutword"] = test_df.text.apply(chinese_pre)
    print(train_df.cutword.head())

    labelMap = {"体育": 0, "娱乐": 1, "家居": 2, "房产": 3, "教育": 4,
                "时尚": 5, "时政": 6, "游戏": 7, "科技": 8, "财经": 9}
    train_df["labelcode"] = train_df["label"].map(labelMap)
    val_df["labelcode"] = val_df["label"].map(labelMap)
    test_df["labelcode"] = test_df["label"].map(labelMap)

    train_df[["labelcode", "cutword"]].to_csv("../DataFrame/THUCNews/cnews_train2.csv", index=False)
    val_df[["labelcode", "cutword"]].to_csv("../DataFrame/THUCNews/cnews_val2.csv", index=False)
    test_df[["labelcode", "cutword"]].to_csv("../DataFrame/THUCNews/cnews_test2.csv", index=False)


def train_model(model, traindataloader, valdataloader, criterion, optimizer, num_epochs=25):
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    since = time.time()
    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        model.train()
        for step, batch in enumerate(traindataloader):
            textdata, target = batch.cutword[0], batch.labelcode.view(-1)
            output = model(textdata)
            pre_lab = torch.argmax(output, 1)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(target)
            train_corrects += torch.sum(pre_lab == target.data)
            train_num +=  len(target)

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))

        model.eval()
        for step, batch in enumerate(valdataloader):
            textdata, target = batch.cutword[0], batch.labelcode.view(-1)
            output = model(textdata)
            pre_lab = torch.argmax(output, 1)
            loss = criterion(output, target)
            val_loss += loss.item() * len(target)
            val_corrects += torch.sum(pre_lab == target.data)
            val_num += len(target)

        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

    train_process = pd.DataFrame(
        data={
            "epoch": range(num_epochs),
            "train_loss_all": train_loss_all,
            "train_acc_all": train_acc_all,
            "val_loss_all": val_loss_all,
            "val_acc_all": val_acc_all
        }
    )

    return model, train_process


def test_model(model, testdataloader):
    model.eval()
    test_y_all = torch.LongTensor()
    pre_lab_all = torch.LongTensor()
    for step, batch in enumerate(testdataloader):
        textdata, target = batch.cutword[0], batch.labelcode.view[-1]
        output = model(textdata)
        pre_lab = torch.argmax(output, 1)
        test_y_all = torch.cat((test_y_all, target))
        pre_lab_all = torch.cat((pre_lab_all, pre_lab))

    acc = accuracy_score(test_y_all, pre_lab_all)
    print("Acc on testing dataset:", acc)

    class_label = {"体育", "娱乐", "家居", "房产", "教育", "时尚", "时政", "游戏", "科技", "财经"}
    conf_mat = confusion_matrix(test_y_all, pre_lab_all)
    df_cm = pd.DataFrame(conf_mat, index=class_label, columns=class_label)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontproperties=fonts)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontproperties=fonts)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


if __name__ == '__main__':
    prepare_dataset()
    mytokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=mytokenize,
                      include_lengths=True, use_vocab=True,
                      batch_first=True, fix_length=400)
    LABEL = data.Field(sequential=False, use_vocab=False,
                       pad_token=None, unk_token=None)

    text_data_fields = [
        ("labelcode", LABEL),
        ("cutword", TEXT)
    ]

    traindata, valdata, testdata = data.TabularDataset.splits(
        path="../DataFrame/THUCNews", format="csv",
        train="cnews_train2.csv", fields=text_data_fields,
        validation="cnews_val2.csv", test="cnews_test2.csv",
        skip_header=True
    )
    print('traindata: ', len(traindata), 'valdata: ', len(valdata), 'testdata: ', len(testdata))

    TEXT.build_vocab(traindata, max_size=20000, vectors=None)
    LABEL.build_vocab(traindata)
    word_fre = TEXT.vocab.freqs.most_common(n=50)
    word_fre = pd.DataFrame(data=word_fre, columns=["word", "fre"])
    word_fre.plot(x="word", y="fre", kind="bar", legend=False, figsize=(12, 7))
    plt.xticks(rotation=90, fontproperties=fonts, size=10)
    plt.show()

    BATCH_SIZE = 64
    train_iter = data.BucketIterator(traindata, batch_size=BATCH_SIZE)
    val_iter = data.BucketIterator(valdata, batch_size=BATCH_SIZE)
    test_iter = data.BucketIterator(testdata, batch_size=BATCH_SIZE)

    vocab_size = len(TEXT.vocab)
    embedding_dim = 100
    hidden_dim = 128
    layer_dim = 1
    output_dim = 10
    lstmmodel = LSTMNet(vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim)
    print(lstmmodel)

    optimizer = optim.Adam(lstmmodel.parameters(), lr=0.0003)
    loss_func = nn.CrossEntropyLoss()
    lstmmodel, train_process = train_model(lstmmodel, train_iter, val_iter, loss_func, optimizer, num_epochs=20)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_process.epoch, train_process.train_loss_all, "r.-", label="Train Loss")
    plt.plot(train_process.epoch, train_process.val_loss_all, "bs-", label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch number", size=13)
    plt.ylabel("Loss value", size=13)
    plt.subplot(1, 2, 2)
    plt.plot(train_process.epoch, train_process.train_acc_all, "r.-", label="Train Acc")
    plt.plot(train_process.epoch, train_process.test_acc_all, "bs-", label="Test Acc")
    plt.xlabel("Epoch number", size=13)
    plt.ylabel("Acc value", size=13)
    plt.show()

    test_model(lstmmodel, test_iter)

    from sklearn.manifold import TSNE
    lstmmodel = torch.load("../Module/MyLSTMNet/lstmmodel.plk")
    word2vec = lstmmodel.embedding.weight
    words = TEXT.vocab.itos
    tsne = TSNE(n_components=2, random_state=123)
    word2vec_tsne = tsne.fit_transform(word2vec.data.numpy())
    plt.figure(figsize=(10, 8))
    plt.scatter(word2vec_tsne[:, 0], word2vec_tsne[:, 1], s=4)
    plt.title("word2vec distribution: ", fontproperties=fonts, size=15)
    plt.show()

    vis_word = ["中国", "市场", "公司", "美国", "记者", "学生", "游戏", "北京",
                "投资", "电影", "银行", "工作", "留学", "大学", "经济", "产品",
                "设计", "方面", "玩家", "学校", "学习", "房价", "专家", "楼市"]

    vis_word_index = [words.index(ii) for ii in vis_word]
    plt.figure(figsize=(10, 8))
    for ii, index in enumerate(vis_word_index):
        plt.autoscale(word2vec_tsne[index, 0], word2vec_tsne[index, 1])
        plt.text(word2vec_tsne[index, 0], word2vec_tsne[index, 1], vis_word[ii], fontproperties=fonts)
        plt.title("word2vec distribution: ", fontproperties=fonts, size=15)
        plt.show()


