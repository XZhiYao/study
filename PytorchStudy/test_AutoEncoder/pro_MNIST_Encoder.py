import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import hiddenlayer as hl
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid


def prepare_dataset():
    train_data = MNIST(
        root='../Dataset/MNIST',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    train_data_x = train_data.data.type(torch.FloatTensor) / 255.0
    train_data_x = train_data_x.reshape(train_data_x.shape[0], -1)
    train_data_y = train_data.targets

    train_loader = Data.DataLoader(
        dataset=train_data_x,
        batch_size=64,
        shuffle=True,
        num_workers=2,
    )

    test_data = MNIST(
        root='../Dataset/MNIST',
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )

    test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
    test_data_x = test_data_x.reshape(test_data_x.shape[0], -1)
    test_data_y = test_data.targets
    print("Training Dataset: ", train_data_x.shape)
    print("Testing Dataset: ", test_data_x.shape)

    return train_loader, test_data_x, test_data_y

if __name__ == '__main__':
    train_loader, test_data_x, test_data_y = prepare_dataset()