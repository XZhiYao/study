import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
from torch.optim import SGD
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import hiddenlayer as hl
from torchviz import make_dot
from tensorboardX import SummaryWriter
import time
from visdom import Visdom

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2,
            ),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(
                in_features=32 * 7 * 7,
                out_features=128,
            ),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        output = self.out(x)
        return output


def prepare_dataset():
    train_data = torchvision.datasets.MNIST(
        root="../Dataset",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=False
    )

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=128,
        shuffle=True,
        num_workers=1,
    )

    test_data = torchvision.datasets.MNIST(
        root="../Dataset",
        train=False,
        download=False
    )

    test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
    test_data_x = torch.unsqueeze(test_data_x, dim=1)
    test_data_y = test_data.targets
    print("test_data_x.shape: ", test_data_x.shape)
    print("test_data_y.shape: ", test_data_y.shape)

    return train_loader, test_data_x, test_data_y


def save_module_img():
    # save network structure as image
    hl_graph = hl.build_graph(MyConvnet, torch.zeros([1, 1, 28, 28]))
    hl_graph.theme = hl.graph.THEMES["blue"].copy()
    hl_graph.save("../Module/Image/MyConvnet_hl.png", format="png")

    # # PyTorchViz
    x = torch.randn(1, 1, 28, 28).requires_grad_(True)
    y = MyConvnet(x)
    MyConvnetvis = make_dot(y, params=dict(list(MyConvnet.named_parameters()) + [('x', x)]))
    MyConvnetvis.format = "png"
    MyConvnetvis.directory = "../Module/MyConvnet_vis"
    MyConvnetvis.view()


def train_tensorboardX(MyConvnet):
    SumWriter = SummaryWriter(logdir="../Log/tensorboardX")
    optimizer = torch.optim.Adam(MyConvnet.parameters(), lr=0.0003)
    loss_func = nn.CrossEntropyLoss()
    train_loss = 0
    print_loss = 100

    for epoch in range(5):
        for step, (b_x, b_y) in enumerate(train_loader):
            output = MyConvnet(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss

            niter = epoch * len(train_loader) + step + 1

            if niter % print_loss == 0:
                SumWriter.add_scalar("train loss", train_loss.item() / niter, global_step=niter)

                output = MyConvnet(test_data_x)
                _, pre_lab = torch.max(output, 1)  # 在第一维度（行）中选出最大的一个值，返回这个值的（行）下标位置
                acc = accuracy_score(test_data_y, pre_lab)
                SumWriter.add_scalar("test acc", acc.item(), niter)

                b_x_im = vutils.make_grid(b_x, nrow=12)
                SumWriter.add_image("train image sample", b_x_im, niter)

                for name, param in MyConvnet.named_parameters():
                    SumWriter.add_histogram(name, param.data.numpy(), niter)


def train_HiddenLayer(MyConvnet):
    optimizer = torch.optim.Adam(MyConvnet.parameters(), lr=0.0003)
    loss_func = nn.CrossEntropyLoss()

    history1 = hl.History()
    canvas1 = hl.Canvas()
    print_step = 100

    for epoch in range(5):
        for step, (b_x, b_y) in enumerate(train_loader):
            output = MyConvnet(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            niter = epoch * len(train_loader) + step + 1
            if niter % print_step == 0:
                output = MyConvnet(test_data_x)
                _, pre_lab = torch.max(output, 1)
                acc = accuracy_score(test_data_y, pre_lab)
                # print('niter: ', niter)
                history1.log(niter,
                             train_loss=loss,
                             test_acc=acc,
                             hidden_weight=MyConvnet.fc[2].weight)

                # print(history1.history[(epoch, niter)]["train_loss"])
                # print(history1.history["({},{})".format(epoch, step)]["train_loss"])
                # print(type(history1["train_loss"]))
                print(history1["train_loss"])
                with canvas1:
                    canvas1.draw_plot(history1["train_loss"])
                    canvas1.draw_plot(history1["test_acc"])
                    canvas1.draw_image(history1["hidden_weight"])


if __name__ == '__main__':
    train_loader, test_data_x, test_data_y = prepare_dataset()
    MyConvnet = ConvNet()
    print(MyConvnet)
    # save_module_img()
    # train_tensorboardX(MyConvnet)
    # train_HiddenLayer(MyConvnet)
    for step, (b_x, b_y) in enumerate(train_loader):
        if step > 0:
            break

    print(b_x.shape)
    print(b_y.shape)

    vis = Visdom()
    vis.image(b_x[0, :, :, :], win="one image", env="MyimagePlot",
              opts=dict(title="a  single image"))

    vis.images(b_x, nrow=16, win="my batch image", env="MyimagePlot",
              opts=dict(title="a batch image"))

    texts = """A flexible tool for creating, organizing, and sharing visualizations of live, rich data.
    Supports Torch and Numpy."""
    vis.text(texts, win="text plot", env="MyimagePlot", opts=dict(title="vision text"))
