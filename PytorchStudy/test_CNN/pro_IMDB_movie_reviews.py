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
from torchtext.legacy import data
from torchtext.vocab import Vectors, GloVe
from sklearn.model_selection import train_test_split


class CNN_Text(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=n_filters,
                kernel_size=(fs, embedding_dim)
            ) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)


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


def view_Negative_Positive_Reviews(train_label, traindata):
    plt.figure(figsize=(16, 8))
    for ii in np.unique(train_label):
        text = np.array(traindata.train_word[traindata.train_label == ii])
        text = " ".join(np.concatenate(text))
        plt.subplot(1, 2, ii + 1)
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


def train_epoch(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    train_corrects = 0
    train_num = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        pre = model(batch.text[0]).squeeze(1)
        loss = criterion(pre, batch.label.type(torch.FloatTensor))
        pre_lab = torch.round(torch.sigmoid(pre))
        train_corrects += torch.sum(pre_lab.long() == batch.label)
        train_num += len(batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / train_num
    epoch_acc = train_corrects.double().item() / train_num

    return epoch_loss, epoch_acc


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    train_corrects = 0
    train_num = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            pre = model(batch.text[0]).squeeze(1)
            loss = criterion(pre, batch.label.type(torch.FloatTensor))
            pre_lab = torch.round(torch.sigmoid(pre))
            train_corrects += torch.sum(pre_lab.long() == batch.label)
            train_num += len(batch.label)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / train_num
        epoch_acc = train_corrects.double().item() / train_num

    return epoch_loss, epoch_acc


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
    # view_Negative_Positive_Reviews(train_label, traindata)

    mytokenize = lambda x: x.split()
    # Defines a general datatype
    TEXT = data.Field(sequential=True, tokenize=mytokenize,
                      include_lengths=True, use_vocab=True,
                      batch_first=True, fix_length=200)
    LABEL = data.Field(sequential=False, use_vocab=False,
                       pad_token=None, unk_token=None)

    train_test_fields = [
        ("label", LABEL),
        ("text", TEXT)
    ]
    traindata, testdata = data.TabularDataset.splits(
        path="../DataFrame/IMDB", format="csv",
        train="imdb_train.csv", fields=train_test_fields,
        test="imdb_test.csv", skip_header=True
    )

    print(len(traindata), len(testdata))
    ex0 = traindata.examples[0]
    print(ex0)
    print(ex0.text)

    train_data, val_data = traindata.split(split_ratio=0.7)
    print(len(train_data), len(val_data))

    vec = Vectors("glove.6B.100d.txt", "../DataFrame/IMDB")
    TEXT.build_vocab(train_data, max_size=20000, vectors=vec)
    LABEL.build_vocab(train_data)

    print(TEXT.vocab.freqs.most_common(n=10))
    print("keyword num: ", len(TEXT.vocab.itos))
    print("10 num:\n", TEXT.vocab.itos[0:10])
    print("classifica label:", LABEL.vocab.freqs)

    BATCH_SIZE = 32
    train_iter = data.BucketIterator(train_data, batch_size=BATCH_SIZE)
    val_iter = data.BucketIterator(val_data, batch_size=BATCH_SIZE)
    test_iter = data.BucketIterator(testdata, batch_size=BATCH_SIZE)

    for step, batch in enumerate(train_iter):
        if step > 0:
            break
    print("classifica label:\n", batch.label)
    print("data size:", batch.text[0].shape)
    print("data sample:", len(batch.text[1]))

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    N_FILTERS = 100
    FILTER_SIZES = [3, 4, 5]
    OUTPUT_DIM = 1
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    model = CNN_Text(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
    print(model)

    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    EPOCHS = 10
    best_val_loss = float("inf")
    best_acc = float(0)
    for epoch in range(EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_iter, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_iter, criterion)
        end_time = time.time()
        print("Epoch: ", epoch+1, "|", "Epoch Time: ", end_time - start_time, "s")
        print("Train Loss: ", train_loss, "|", "Train Acc: ", train_acc)
        print("Val. Loss: ", val_loss, "|", "Val. Acc: ", val_acc)

        if (val_loss < best_val_loss) & (val_acc > best_acc):
            best_model_wts = copy.deepcopy(model.state_dict())
            best_val_loss = val_loss
            best_acc = val_acc

    model.load_state_dict(best_model_wts)

    test_loss, test_acc = evaluate(model, test_iter, criterion)
    print("acc on testing dataset: ", test_acc)




