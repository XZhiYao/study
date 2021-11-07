import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.datasets import fetch_california_housing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data as Data
import matplotlib.pyplot as plt
import seaborn as sns


class MLPregression(nn.Module):
    def __init__(self):
        super(MLPregression, self).__init__()
        self.hidden1 = nn.Linear(
            in_features=8,
            out_features=100,
            # out_features=16,
            bias=True
        )

        self.hidden2 = nn.Linear(100, 100)
        self.hidden3 = nn.Linear(100, 50)
        self.predict = nn.Linear(50, 1)
        # self.hidden3 = nn.Linear(16, 4)
        # self.predict = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        output = self.predict(x)

        return output[:, 0]


def view_corelationship(housedata_df):
    datacor = np.corrcoef(housedata_df.values, rowvar=0)
    datacor = pd.DataFrame(data=datacor, columns=housedata_df.columns, index=housedata_df.columns)
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(datacor, square=True, annot=True, fmt=".3f", linewidths=.5, cmap="YlGnBu",
                     cbar_kws={"fraction": 0.046, "pad": 0.03})
    plt.show()


def prepare_dataset():
    housedata = fetch_california_housing()
    print('housedata:\n', len(housedata.data), len(housedata.target), type(housedata.data), type(housedata.target))
    X_train, X_test, y_train, y_test = train_test_split(housedata.data, housedata.target, test_size=0.3,
                                                        random_state=42)
    scale = StandardScaler()
    X_train_s = scale.fit_transform(X_train)
    X_test_s = scale.transform(X_test)

    housedata_df = pd.DataFrame(data=X_train_s, columns=housedata.feature_names)
    housedata_df['target'] = y_train
    print('housedata_df:\n', len(housedata_df.values))
    # view_corelationship(housedata_df)
    train_xt = torch.from_numpy(X_train_s.astype(np.float32))
    train_yt = torch.from_numpy(y_train.astype(np.float32))
    test_xt = torch.from_numpy(X_test_s.astype(np.float32))
    test_yt = torch.from_numpy(y_test.astype(np.float32))

    train_data = Data.TensorDataset(train_xt, train_yt)

    test_data = Data.TensorDataset(test_xt, test_yt)
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=64,
        shuffle=True,
        num_workers=1
    )

    return train_loader, test_xt, test_yt


def view_loss(train_loss_all):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_all, "ro-", label="Train Loss")
    plt.legend()
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.show()


if __name__ == '__main__':
    train_loader, test_xt, test_yt = prepare_dataset()
    mlpreg = MLPregression()
    print(mlpreg)
    optimizer = torch.optim.SGD(mlpreg.parameters(), lr=0.01)
    loss_func = nn.MSELoss()
    train_loss_all = []

    for epoch in range(30):
        train_loss = 0
        train_num = 0
        for step, (b_x, b_y) in enumerate(train_loader):
            output = mlpreg(b_x)
            # print('output:\n', output)
            loss = loss_func(output, b_y)
            # print('loss:\n', loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('loss.item(): ', loss.item())
            train_loss += loss.item() * b_x.size(0)
            train_num += b_x.size(0)

        train_loss_all.append(train_loss / train_num)

    # view_loss(train_loss_all)
    pre_y = mlpreg(test_xt)
    pre_y = pre_y.data.numpy()
    mae = mean_absolute_error(test_yt, pre_y)
    print('mae: ', mae)
    index = np.argsort(test_yt)
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(len(test_yt)), test_yt[index], "r", label="Original Y")
    plt.scatter(np.arange(len(pre_y)), pre_y[index], s=3, c="b", label="Prediction")
    plt.legend(loc="upper left")
    plt.grid()
    plt.xlabel("Index")
    plt.ylabel("Y")
    plt.show()
