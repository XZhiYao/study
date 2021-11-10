import torch
import torch.nn as nn
from torch.optim import SGD
import torch.utils.data as Data
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def prepare_dataset():
    # read data
    boston_X, boston_y = load_boston(return_X_y=True)
    print("boston_X.shape: ", boston_X.shape)
    # plt.figure()
    # plt.hist(boston_y, bins=20)
    # plt.show()
    # Dataload
    ss = StandardScaler(with_mean=True, with_std=True)
    boston_Xs = ss.fit_transform(boston_X)
    train_xt = torch.from_numpy(boston_Xs.astype(np.float32))
    train_yt = torch.from_numpy(boston_y.astype(np.float32))
    train_data = Data.TensorDataset(train_xt, train_yt)
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=128,
        shuffle=True,
        num_workers=0,
    )
    return train_loader


# define Method: Module
class MLPmodel(nn.Module):
    def __init__(self):
        super(MLPmodel, self).__init__()
        self.hidden1 = nn.Linear(
            in_features=13,
            out_features=10,
            bias=True,
        )
        self.active1 = nn.ReLU()
        self.hidden2 = nn.Linear(10, 10)
        self.active2 = nn.ReLU()
        self.regression = nn.Linear(10, 1)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.active1(x)
        x = self.hidden2(x)
        x = self.active2(x)
        output = self.regression(x)
        return output


def train(train_loader, mlp, optimizer, loss_func):

    train_loss_all = []
    for epoch in range(30):
        for step, (b_x, b_y) in enumerate(train_loader):
            output = mlp(b_x).flatten()
            train_loss = loss_func(output, b_y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_loss_all.append(train_loss.item())
    return train_loss_all


# define Method: Sequential
class MLPmodel2(nn.Module):
    def __init__(self):
        super(MLPmodel2, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(13, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
        )
        self.regression = nn.Linear(10, 1)

    def forward(self, x):
        x = self.hidden(x)
        output = self.regression(x)
        return output


if __name__ == '__main__':
    train_loader = prepare_dataset()
    mlp1 = MLPmodel()
    mlp2 = MLPmodel2()
    print(mlp1.parameters())
    optimizer1 = SGD(mlp1.parameters(), lr=0.001)
    optimizer2 = SGD(mlp2.parameters(), lr=0.001)
    loss_func = nn.MSELoss()
    # train_loss_all = train(train_loader, mlp1, optimizer1, loss_func)
    train_loss_all = train(train_loader, mlp2, optimizer2, loss_func)
    print(len(train_loss_all))
    # plt.figure()
    # plt.plot(train_loss_all, "r-")
    # plt.title("Train loss per iteration")
    # plt.show()

    # save module structure
    torch.save(mlp2, "../Module/mlp2.pkl")
    mlp2load = torch.load("../Module/mlp2.pkl")
    print(mlp2load)

    # just save module parameter
    torch.save(mlp2.state_dict(), "../Module/mlp2_param.pkl")
    mlp2param = torch.load("../Module/mlp2_param.pkl")
    print(mlp2param)


