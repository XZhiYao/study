import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import copy
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data as Data
from torchvision import transforms
import hiddenlayer as hl


class RNNimc(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        """
        hidden_dim: RNN neurons num
        layer_dim: RNN layer num
        """
        super(RNNimc, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.fc1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, h_n = self.rnn(x, None)
        output = self.fc1(output[:, -1, :])
        return output


def prepare_dataset():
    train_data = torchvision.datasets.MNIST(
        root="../Dataset",
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=64,
        shuffle=True,
        num_workers=2
    )

    test_data = torchvision.datasets.MNIST(
        root="../Dataset",
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )

    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=64,
        shuffle=True,
        num_workers=2
    )
    return train_loader, test_loader


def save_module_img(MyRNNimc):
    hl_graph = hl.build_graph(MyRNNimc, torch.zeros([1, 28, 28]))
    hl_graph.theme = hl.graph.THEMES["blue"].copy()
    hl_graph.save("../Module/MyRNNimc_hl.png", format="png")


if __name__ == '__main__':
    train_loader, test_loader = prepare_dataset()

    input_dim = 28
    hidden_dim = 128
    layer_dim = 1
    output_dim = 10
    MyRNNimc = RNNimc(input_dim, hidden_dim, layer_dim, output_dim)
    print(MyRNNimc)
    # save_module_img(MyRNNimc)

    optimizer = torch.optim.RMSprop(MyRNNimc.parameters(), lr=0.0003)
    criterion = nn.CrossEntropyLoss()
    train_loss_all = []
    train_acc_all = []
    test_loss_all = []
    test_acc_all = []
    num_epochs = 30
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        MyRNNimc.train()
        corrects = 0
        loss_count = 0
        train_num = 0
        for step, (b_x, b_y) in enumerate(train_loader):
            xdata = b_x.view(-1, 28, 28)
            output = MyRNNimc(xdata)
            pre_lab = torch.argmax(output, 1)
            loss = criterion(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_count += loss.item() * b_x.size(0)
            corrects += torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)
            # print('parten loss: ', loss)

        train_loss_all.append(loss_count / train_num)
        train_acc_all.append(corrects.double().item() / train_num)
        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))

        MyRNNimc.eval()
        corrects = 0
        test_num = 0
        for step, (b_x, b_y) in enumerate(test_loader):
            xdata = b_x.view(-1, 28, 28)
            output = MyRNNimc(xdata)
            pre_lab = torch.argmax(output, 1)
            loss = criterion(output, b_y)
            loss_count += loss.item() * b_x.size(0)
            corrects += torch.sum(pre_lab == b_y.data)
            test_num += b_x.size(0)

        test_loss_all.append(loss_count / test_num)
        test_acc_all.append(corrects.double().item() / test_num)
        print('{} Test Loss: {:.4f} Test Acc: {:.4f}'.format(epoch, test_loss_all[-1], test_acc_all[-1]))

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_all, "ro-", label="Train Loss")
    plt.plot(test_loss_all, "bs-", label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_all, "ro-", label="Train Acc")
    plt.plot(test_acc_all, "bs-", label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.legend()
    plt.show()
