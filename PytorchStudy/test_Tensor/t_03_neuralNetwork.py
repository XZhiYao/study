import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 输入图像channel：1；输出channel：6；5x5卷积核
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 2x2 Max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果是方阵,则可以只使用一个数字进行定义
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def t_num_features(net):
    test = torch.rand(16, 5, 5)
    print(test)
    print(test.size())
    size = test.size()[1:]
    print(size)
    num_features = 1
    for s in size:
        print(s)
        num_features *= s
    print('num_features:', num_features)

    params = list(net.parameters())
    print(len(params))
    print(params[0].size())


def t_dnn_loss(net):
    input = torch.randn(1, 1, 32, 32)
    print(input.shape)
    output = net(input)
    print(output.size(), output)
    net.zero_grad()

    output.backward(torch.randn(1, 10), retain_graph=True)

    target = torch.randn(10)  # 本例子中使用模拟数据
    print(target.size())
    target = target.view(1, -1)  # 使目标值与数据值尺寸一致
    print(target.size())
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    print('loss:', loss)
    # calculate map:
    # input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
    # -> view -> linear -> relu -> linear -> relu -> linear
    # -> MSELoss
    # -> loss
    print(loss.grad_fn)  # MSELoss
    print(loss.grad_fn.next_functions[0][0])  # Linear
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

    net.zero_grad()  # 清零所有参数(parameter）的梯度缓存

    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)

    loss.backward()

    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)

    return output, loss


def t_dnn_update_weight(net):
    learning_rate = 0.01
    for f in net.parameters():
        print(f.size())
        f.data.sub_(f.grad.data * learning_rate)

    # 创建优化器(optimizer）
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    optimizer.zero_grad()

    input = torch.randn(1, 1, 32, 32)
    output = net(input)
    target = torch.randn(10)
    target = target.view(1, -1)
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()  # 更新参数
    for f in net.parameters():
        print(f.grad.data)


if __name__ == '__main__':
    net = Net()
    print(net)
    output, loss = t_dnn_loss(net)
    t_dnn_update_weight(net)
