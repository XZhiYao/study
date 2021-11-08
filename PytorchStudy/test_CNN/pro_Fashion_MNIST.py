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


if __name__ == '__main__':
    train_data = FashionMNIST(
        root="../Dataset",
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )
    test_data = FashionMNIST(
        root="../Dataset",
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=64,
        shuffle=False,
        num_workers=2,
    )
    print("The train_loader num of batch are: ", len(train_loader))

    for step, (b_x, b_y) in enumerate(train_loader):
        if step > 0:
            break
    batch_x = b_x.squeeze().numpy()
    batch_y = b_y.numpy()
    class_label = train_data.classes
    class_label[0] = "T-shirt"
    plt.figure(figsize=(12, 5))
    for ii in np.arange(len(batch_y)):
        plt.subplot(4, 16, ii+1)
        plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
        plt.title(class_label[batch_y[ii]], size=9)
        plt.axis("off")
        plt.subplots_adjust(wspace=0.05)
    plt.show()