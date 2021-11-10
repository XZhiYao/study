"""
Test Module: 1.LeNet5 2. Dilation LaNet5
Dataset: FashionMNIST
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import time
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST


class MyLeNet5(nn.Module):
    def __init__(self):
        super(MyLeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            # (1, 28, 28) -> (16, 28, 28)
            nn.ReLU(),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2,
            ),
            # (16, 28, 28) -> (16, 14, 14)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            # (16, 14, 14) -> (32, 12, 12)
            nn.ReLU(),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2,
            ),
            # (32, 12, 12) -> (32, 6, 6)
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=32 * 6 * 6,
                out_features=256,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=256,
                out_features=128,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=128,
                out_features=10,
            )
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output


class MyDilaLeNet5(nn.Module):
    def __init__(self):
        super(MyDilaLeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=2,
            ),
            # (1, 28, 28) -> (16, 13, 13)
            nn.ReLU(),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2,
            ),
            # (16, 26, 26) -> (16, 13, 13)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=0,
                dilation=2,
            ),
            # (16, 13, 13) -> (32, 9, 9)
            nn.ReLU(),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2,
            ),
            # (32, 9, 9) -> (32, 4, 4)
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=32*4*4,
                out_features=256,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=256,
                out_features=128,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=128,
                out_features=10,
            )
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output


def view_dataset_batch(class_label, train_loader):
    for step, (b_x, b_y) in enumerate(train_loader):
        if step > 0:
            break
    batch_x = b_x.squeeze().numpy()
    batch_y = b_y.numpy()

    plt.figure(figsize=(12, 5))
    for ii in np.arange(len(batch_y)):
        plt.subplot(4, 16, ii + 1)
        plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
        plt.title(class_label[batch_y[ii]], size=9)
        plt.axis("off")
        plt.subplots_adjust(wspace=0.05)
    plt.show()


def view_loss_and_acc(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process.epoch, train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process.epoch, train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process.epoch, train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process.epoch, train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.legend()

    plt.show()


def view_confusion_matrix(test_data_y, pre_lab, class_label):
    conf_mat = confusion_matrix(test_data_y, pre_lab)
    df_cm = pd.DataFrame(conf_mat, index=class_label, columns=class_label)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()


def prepare_dataset():
    train_data = FashionMNIST(
        root="../Dataset/FashionMNIST",
        train=True,
        transform=transforms.ToTensor(),
        download=False
    )
    test_data = FashionMNIST(
        root="../Dataset/FashionMNIST",
        train=False,
        transform=transforms.ToTensor(),
        download=False
    )

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=64,
        shuffle=False,
        num_workers=2,
    )
    print("The train_loader num of batch are: ", len(train_loader))

    class_label = train_data.classes
    class_label[0] = "T-shirt"  # change class_label[0]: ('T-shirt/top') -> ('T-shirt')
    # print('class_label: ', class_label)
    # view_dataset_batch(class_label, train_loader)

    return train_loader, test_data, class_label


def train_model(model, traindataloader, train_rate, criterion, optimizer, num_epochs=25):
    """
    model: Network Model(nn.torch)
    traindataloader: training dataset
    train_rate: training dataset batchsize
    criterion: loss function
    optimizer: optimize methood
    num_epochs: training epoch
    """

    batch_num = len(traindataloader)
    train_batch_num = round(batch_num * train_rate)
    # save model parameter
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        for step, (b_x, b_y) in enumerate(traindataloader):
            if step < train_batch_num:
                model.train()
                output = model(b_x)
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * b_x.size(0)
                train_corrects += torch.sum(pre_lab == b_y.data)
                train_num += b_x.size(0)
            else:
                model.eval()
                output = model(b_x)
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, b_y)
                val_loss += loss.item() * b_x.size(0)
                val_corrects += torch.sum(pre_lab == b_y.data)
                val_num += b_x.size(0)

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # copy the best acc parameter
        # print('epoch {} val_loss_all[-1]: {} and last best_acc: {}'.format(epoch, val_loss_all[-1], best_acc))
        if val_loss_all[-1] > best_acc:
            best_acc = val_loss_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        time_use = time.time() - since
        print("Train and val complete in {:.0f}m {:.0f}".format(time_use // 60, time_use % 60))

    model.load_state_dict(best_model_wts)
    train_process = pd.DataFrame(
        data={
            "epoch": range(num_epochs),
            "train_loss_all": train_loss_all,
            "val_loss_all": val_loss_all,
            "train_acc_all": train_acc_all,
            "val_acc_all": val_acc_all
        }
    )
    return model, train_process


if __name__ == '__main__':
    train_loader, test_data, class_label = prepare_dataset()
    test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
    test_data_x = torch.unsqueeze(test_data_x, dim=1)
    test_data_y = test_data.targets
    print("test_data_x.shape: ", test_data_x.shape)
    print("test_data_y.shape: ", test_data_y.shape)

    # mylenet5 = MyLeNet5()
    # print(mylenet5)

    # optimizer = torch.optim.Adam(mylenet5.parameters(), lr=0.0003)
    # criterion = nn.CrossEntropyLoss()
    # mylenet5, train_process = train_model(mylenet5, train_loader, 0.8, criterion, optimizer, num_epochs=25)
    #
    # # view_MyLeNet5_loss_and_acc(train_process)
    #
    # mylenet5.eval()
    # output = mylenet5(test_data_x)
    # pre_lab = torch.argmax(output, 1)
    # acc = accuracy_score(test_data_y, pre_lab)
    # print("The predict acc on testing dataset: ", acc)
    #
    # # view_MyLeNet5_confusion_matrix(test_data_y, pre_lab, class_label)

    mydilalenet5 = MyDilaLeNet5()
    print(mydilalenet5)

    optimizer = torch.optim.Adam(mydilalenet5.parameters(), lr=0.0003)
    criterion = nn.CrossEntropyLoss()

    mydilalenet5, train_process = train_model(mydilalenet5, train_loader, 0.8, criterion, optimizer, num_epochs=25)
    view_loss_and_acc(train_process)

    mydilalenet5.eval()
    output = mydilalenet5(test_data_x)
    pre_lab = torch.argmax(output, 1)
    acc = accuracy_score(test_data_y, pre_lab)
    print("The predict acc on testing dataset: ", acc)

    view_confusion_matrix(test_data_y, pre_lab, class_label)




