"""
Test Tool: Visdom-pytorch
Dataset: UCI-Iris
"""
import torch
import numpy as np
from visdom import Visdom
from sklearn.datasets import load_iris

if __name__ == '__main__':
    iris_x, iris_y = load_iris(return_X_y=True)
    print(iris_x.shape)
    print(iris_y.shape)
    # rum python -m visdom.server in Terminal to connect server(http://localhost:8097)
    vis = Visdom()
    # 2D
    vis.scatter(iris_x[:, 0:2], Y=iris_y+1, win="windows_2D", env="main")
    # 3D
    vis.scatter(iris_x[:, 0:3], Y=iris_y+1, win="windows_3D", env="main",
                opts=dict(markersize=4,
                          xlabel="feature_1",
                          ylabel="feature_2")
                )
    # line plot
    x = torch.linspace(-6, 6, 100).view((-1, 1))
    sigmoid = torch.nn.Sigmoid()
    sigmoidy = sigmoid(x)
    tanh = torch.nn.Tanh()
    tanhy = tanh(x)
    relu = torch.nn.ReLU()
    reluy = relu(x)

    ploty = torch.cat((sigmoidy, tanhy, reluy), dim=1)
    plotx = torch.cat((x, x, x), dim=1)
    vis.line(Y=ploty, X=plotx, win="line plot", env="main",
             opts=dict(dash=np.array(["solid", "dash", "dashdot"]),
                       legend=["Sigmoid", "Tanh", "ReLU"]))

    # stem plot
    x = torch.linspace(-6, 6, 100).view((-1, 1))
    y1 = torch.sin(x)
    y2 = torch.cos(x)

    plotx = torch.cat((y1, y2), dim=1)
    ploty = torch.cat((x, x), dim=1)
    vis.stem(X=plotx, Y=ploty, win="stem plot", env="main",
             opts=dict(legend=["sin", "cos"],
                       title="stems and leaves"))

    # heatmap
    iris_corr = torch.from_numpy(np.corrcoef(iris_x, rowvar=False))
    vis.heatmap(iris_corr, win="heatmap", env="main",
                opts=dict(rownames=["x1", "x2", "x3", "x4"],
                          columnnames=["x1", "x2", "x3", "x4"],
                          title="heatmap"))