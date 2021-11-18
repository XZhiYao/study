import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time
import copy
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchtext.legacy import data
from torchtext.vocab import Vectors


class GRUNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        embeds = self.embedding(x)
        r_out, h_n = self.gru(embeds, None)
        out = self.fc1(r_out[:, -1, :])
        return out


def train_model(model, traindataloader, testdataloader, criterion, optimizer, num_epoch=25):
    train_loss_all = []
    train_acc_all = []
    test_loss_all = []
    test_acc_all = []
    learn_rate = []
    since = time.time()
    schedular = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(num_epoch):
        learn_rate.append(schedular.get_last_lr()[0])
        print('-' * 10)
        print('Epoch {}/{}, Lr:{}'.format(epoch, num_epoch-1, learn_rate[-1]))
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        test_loss = 0.0
        test_corrects = 0
        test_num = 0
        model.train()
        for step, batch in enumerate(traindataloader):
            textdata, target = batch.text[0], batch.label
            output = model(textdata)
            pre_lab = torch.argmax(output, 1)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(target)
            train_corrects += torch.sum(pre_lab == target.data)
            train_num += len(target)

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))

        model.eval()
        for step, batch in enumerate(testdataloader):
            textdata, target = batch.text[0], batch.label
            output = model(textdata)
            pre_lab = torch.argmax(output, 1)
            loss = criterion(output, target)
            test_loss += loss.item() * len(target)
            test_corrects += torch.sum(pre_lab == target.data)
            test_num += len(target)

        test_loss_all.append(test_loss / test_num)
        test_acc_all.append(test_corrects.double().item() / test_num)
        print('{} Test Loss: {:.4f} Test Acc: {:.4f}'.format(epoch, test_loss_all[-1], test_acc_all[-1]))

    train_process = pd.DataFrame(
        data={
            "epoch": range(num_epoch),
            "train_loss_all": train_loss_all,
            "train_acc_all": train_acc_all,
            "test_loss_all": test_loss_all,
            "test_acc_all": test_acc_all,
            "learn_rate": learn_rate
        }
    )

    return model, train_process


if __name__ == '__main__':
    mytokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=mytokenize,
                      include_lengths=True, use_vocab=True,
                      batch_first=True, fix_length=200)
    LABEL = data.Field(sequential=False, use_vocab=False,
                       pad_token=None, unk_token=None)

    train_test_fields = [
        ("text", TEXT),
        ("label", LABEL)
    ]

    traindata, testdata = data.TabularDataset.splits(
        path="../DataFrame/IMDB", format="csv",
        train="imdb_train.csv", fields=train_test_fields,
        test="imdb_test.csv", skip_header=True
    )

    vec = Vectors("glove.6B.100d.txt", "../DataFrame/IMDB")
    TEXT.build_vocab(traindata, max_size=20000, vectors=vec)
    LABEL.build_vocab(traindata)
    BATCH_SIZE = 32
    train_iter = data.BucketIterator(traindata, batch_size=BATCH_SIZE)
    test_iter = data.BucketIterator(testdata, batch_size=BATCH_SIZE)

    vocab_size = len(TEXT.vocab)
    embedding_dim = vec.dim
    hidden_dim = 128
    layer_dim = 1
    output_dim = 2
    grumodel = GRUNet(vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim)
    print(grumodel)

    grumodel.embedding.weight.data.copy_(TEXT.vocab.vectors)
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    grumodel.embedding.weight.data[UNK_IDX] = torch.zeros(vec.dim)
    grumodel.embedding.weight.data[PAD_IDX] = torch.zeros(vec.dim)

    optimizer = optim.RMSprop(grumodel.parameters(), lr=0.003)
    loss_func = nn.CrossEntropyLoss()
    grumodel, train_proccess = train_model(grumodel, train_iter, test_iter, loss_func, optimizer, num_epoch=10)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_proccess.epoch, train_proccess.train_loss_all, "r.-", label="Train Loss")
    plt.plot(train_proccess.epoch, train_proccess.test_loss_all, "bs-", label="Test Loss")
    plt.legend()
    plt.xlabel("Epoch Number", size=13)
    plt.ylabel("Loss Value:", size=13)
    plt.subplot(1, 2, 2)
    plt.plot(train_proccess.epoch, train_proccess.train_acc_all, "r.-", label="Test Acc")
    plt.plot(train_proccess.epoch, train_proccess.test_acc_all, "bs-", label="Test Acc")
    plt.xlabel("Epoch Number", size=13)
    plt.ylabel("Acc", size=13)
    plt.legend()
    plt.show()

    # Testing
    grumodel.eval()
    test_y_all = torch.LongTensor()
    pre_lab_all = torch.LongTensor()
    for step, batch in enumerate(test_iter):
        textdata, target = batch.text[0], batch.label.view(-1)
        output = grumodel(textdata)
        pre_lab = torch.argmax(output, 1)
        test_y_all = torch.cat((test_y_all, target))
        pre_lab_all = torch.cat((pre_lab_all, pre_lab))

    acc = accuracy_score(test_y_all, pre_lab_all)
    print("Acc on Testing Dataset: ", acc)


