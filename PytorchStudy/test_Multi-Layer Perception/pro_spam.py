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


class MLPclassifica(nn.Module):
    def __init__(self):
        super(MLPclassifica, self).__init__()

        self.hidden1 = nn.Sequential(
            nn.Linear(
                in_features=57,
                out_features=30,
                bias=True,
            ),
            nn.ReLU()
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(30, 10),
            nn.ReLU()
        )

        self.classifica = nn.Sequential(
            nn.Linear(10, 2),
            nn.ReLU()
        )

    def forward(self, x):
        fc1 = self.hidden1(x)
        fc2 = self.hidden2(fc1)
        output = self.classifica(fc2)

        return fc1, fc2, output


def prepare_dataset():
    spam = pd.read_csv("../Dataset/Spambase/spambase.csv")
    # print(spam.head())

    spam.columns = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our',
                    'word_freq_over',
                    'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail', 'word_freq_receive',
                    'word_freq_will',
                    'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
                    'word_freq_business', 'word_freq_email',
                    'word_freq_you', 'word_freq_credit', 'word_freq_your', 'word_freq_font', 'word_freq_000',
                    'word_freq_money',
                    'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab',
                    'word_freq_labs',
                    'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85',
                    'word_freq_technology',
                    'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct', 'word_freq_cs',
                    'word_freq_meeting',
                    'word_freq_original', 'word_freq_project', 'word_freq_re', 'word_freq_edu', 'word_freq_table',
                    'word_freq_conference',
                    'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#',
                    'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'label']
    # patten = 'word_'
    # re.match(patten, f)
    print(spam.head())
    print(pd.value_counts(spam.label))
    X = spam.iloc[:, 0:57].values
    y = spam.label.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

    scales = MinMaxScaler(feature_range=(0, 1))
    X_train_s = scales.fit_transform(X_train)
    X_test_s = scales.transform(X_test)

    # colname = spam.columns.values[:-1]
    # plt.figure(figsize=(20, 14))
    # for ii in range(len(colname)):
    #     plt.subplot(7, 9, ii + 1)
    #     sns.boxplot(x=y_train, y=X_train_s[:, ii])
    #     plt.title(colname[ii])
    #
    # plt.subplots_adjust(hspace=0.4)
    # plt.show()
    return X_train, X_test, y_train, y_test


def view_module(mlpc):
    x = torch.randn(1, 57).requires_grad_(True)
    y = mlpc(x)
    Mymlpcvis = make_dot(y, params=dict(list(mlpc.named_parameters()) + [('x', x)]))
    print('Mymlpcvis:\n', Mymlpcvis)
    Mymlpcvis.format = "png"
    Mymlpcvis.directory = "../Module/Mymlpcvis"
    Mymlpcvis.view()


def view_fc2_farward(test_fc2, y_test, mode="forward"):
    if mode == "forward":
        test_fc2_tsne = TSNE(n_components=2).fit_transform(test_fc2.data.numpy())
        print('test_fc2_tsne.shape: ', test_fc2_tsne.shape)
        print(min(test_fc2_tsne[:, 0] - 1), min(test_fc2_tsne[:, 0]), max(test_fc2_tsne[:, 0]))
        plt.figure(figsize=(8, 6))
        plt.xlim([min(test_fc2_tsne[:, 0] - 1), max(test_fc2_tsne[:, 0]) + 1])
        plt.ylim([min(test_fc2_tsne[:, 1] - 1), max(test_fc2_tsne[:, 1]) + 1])
        plt.plot(test_fc2_tsne[y_test == 0, 0], test_fc2_tsne[y_test == 0, 1], "bo", label="0")
        plt.plot(test_fc2_tsne[y_test == 1, 1], test_fc2_tsne[y_test == 1, 0], "rd", label="1")
        plt.legend()
        plt.title("test_fc2_tsne")
        plt.show()

    if mode == "hook":
        classifica = activation["classifica"].data.numpy()
        print("classifica.shape: ", classifica.shape)
        plt.figure(figsize=(8, 6))
        plt.plot(classifica[y_test == 0, 0], classifica[y_test == 0, 1], "bo", label="0")
        plt.plot(classifica[y_test == 1, 0], classifica[y_test == 1, 1], "rd", label="1")
        plt.legend()
        plt.title("classifica")
        plt.show()



activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = prepare_dataset()
    mlpc = MLPclassifica()

    scales = MinMaxScaler(feature_range=(0, 1))
    X_train_s = scales.fit_transform(X_train)
    X_test_s = scales.transform(X_test)

    # X_train_nots = torch.from_numpy(X_train.astype(np.float32))
    X_train_nots = torch.from_numpy(X_train_s.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    # X_test_nots = torch.from_numpy(X_test.astype(np.float32))
    X_test_nots = torch.from_numpy(X_test_s.astype(np.float32))
    y_test_t = torch.from_numpy(y_test.astype(np.int64))
    train_data_nots = Data.TensorDataset(X_train_nots, y_train_t)
    train_nots_loader = Data.DataLoader(
        dataset=train_data_nots,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    optimizer = torch.optim.Adam(mlpc.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()
    historyl = hl.History()
    canvasl = hl.Canvas()
    print_step = 300

    for epoch in range(15):
        for step, (b_x, b_y) in enumerate(train_nots_loader):
            _, _, output = mlpc(b_x)
            train_loss = loss_func(output, b_y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            niter = epoch * len(train_nots_loader) + step + 1
            if niter % print_step == 0:
                mlpc.classifica.register_forward_hook(get_activation("classifica"))
                _, test_fc2, output = mlpc(X_test_nots)
                print("test_fc2.shape: ", test_fc2.shape, type(test_fc2))
                _, pre_lab = torch.max(output, 1)
                # print('pre_lab:\n', pre_lab)
                test_accuracy = accuracy_score(y_test, pre_lab)
                historyl.log(niter, train_loss=train_loss, test_accuracy=test_accuracy)
                view_fc2_farward(test_fc2, y_test, mode="hook")


                # with canvasl:
                #     canvasl.draw_plot(historyl["train_loss"])
                #     canvasl.draw_plot(historyl["test_accuracy"])

