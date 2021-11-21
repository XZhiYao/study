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
from matplotlib import cm


class EnDecoder(nn.Module):
    def __init__(self):
        super(EnDecoder, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 3),
            nn.Tanh(),
        )

        self.Decoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 784),
            nn.Sigmoid(),
        )

    def forward(self, batch_input):
        encoder = self.Encoder(batch_input)
        decoder = self.Decoder(encoder)
        return encoder, decoder


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


def view_single_batch(train_loader):
    for step, b_x in enumerate(train_loader):
        if step > 0:
            break
    im = make_grid(b_x.reshape((-1, 1, 28, 28)))
    im = im.data.numpy().transpose((1, 2, 0))
    plt.figure()
    plt.imshow(im)
    plt.axis("off")
    plt.show()


def view_data_rebuild(edmodel, test_data_x):
    edmodel.eval()
    _, test_decoder = edmodel(test_data_x[0:100, :])
    # plt.figure(figsize=(6, 6))
    # for ii in range(test_decoder.shape[0]):
    #     plt.subplot(10, 10, ii + 1)
    #     im = test_data_x[ii, :]
    #     im = im.data.numpy().reshape(28, 28)
    #     plt.imshow(im, cmap=plt.cm.gray)
    #     plt.axis("off")
    # plt.show()
    plt.figure(figsize=(6, 6))
    for ii in range(test_decoder.shape[0]):
        plt.subplot(10, 10, ii + 1)
        im = test_decoder[ii, :]
        im = im.data.numpy().reshape(28, 28)
        plt.imshow(im, cmap=plt.cm.gray)
        plt.axis("off")
    plt.show()


if __name__ == '__main__':
    train_loader, test_data_x, test_data_y = prepare_dataset()
    # view_single_batch(train_loader)
    edmodel = EnDecoder()
    print(edmodel)

    optimizer = torch.optim.Adam(edmodel.parameters(), lr=0.003)
    loss_func = nn.MSELoss()
    historyl = hl.History()
    canvasl = hl.Canvas()
    train_num = 0
    val_num = 0
    for epoch in range(10):
        train_loss_epoch = 0
        edmodel.train()
        for step, b_x in enumerate(train_loader):
            # print('b_x.shape: ', b_x.shape)
            _, output = edmodel(b_x)
            loss = loss_func(output, b_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item() * b_x.size(0)
            train_num += b_x.size(0)

        train_loss = train_loss_epoch / train_num
        historyl.log(epoch, train_loss=train_loss)

        # with canvasl:
        #     canvasl.draw_plot(historyl["train_loss"])

    # view_data_rebuild(edmodel, test_data_x)

    edmodel.eval()
    TEST_num = 500
    test_encoder, _ = edmodel(test_data_x[0: TEST_num, :])
    print("test_encoder.shape: ", test_encoder.shape)

    # %config InlineBackend.print_figure_kwargs = ['bbox_inches': None]
    test_encoder_arr = test_encoder.data.numpy()
    fig = plt.figure(figsize=(12, 8))
    axl = Axes3D(fig)
    X = test_encoder_arr[:, 0]
    Y = test_encoder_arr[:, 1]
    Z = test_encoder_arr[:, 2]
    axl.set_xlim([min(X), max(X)])
    axl.set_ylim([min(Y), max(Y)])
    axl.set_zlim([min(Z), max(Z)])
    for ii in range(test_encoder.shape[0]):
        text = test_data_y.data.numpy()[ii]
        axl.text(X[ii], Y[ii], Z[ii], str(text), fontsize=8, bbox=dict(boxstyle="round", facecolor=plt.cm.Set1(text), alpha=0.7))
    plt.show()
