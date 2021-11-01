import torch


class TestNet(torch.nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        # self.hidden = torch.nn.Sequential(torch.nn.Linear(13, 10), torch.nn.ReLU(), )
        # self.regression = torch.nn.Linear(10, 1)
        self.conv1 = torch.nn.Conv2d(3, 16, 3)
        self.hidden = torch.nn.Sequential(
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
        )
        self.cla = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.shape[0], -1)
        x = self.hidden(x)
        # output = self.regression(x)
        output = self.cla(x)
        return output


def init_weights(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.normal(m.weight, mean=0, std=0.5)
    if type(m) == torch.nn.Linear:
        torch.nn.init.uniform(m.weight, a=-0.1, b=0.1)
        m.bias.data.fill_(0.01)

testnet = TestNet()
print(testnet)
torch.manual_seed(13)
testnet.apply(init_weights)
# optimizer = torch.optim.Adam(testnet.parameters(), lr=0.001)
# optimizer = torch.optim.Adam([{"params": testnet.hidden.parameters(), "lr": 0.0001},
#                               {"params": testnet.regression.parameters(), "lr": 0.01}], lr=1e-2)
# for input, target in dataset:
#     optimizer.zero_grad()
#     output = testnet(input)
#     loss = loss_fn(output, target)
#     loss.backward()
#     optimizer.step()

# test torch.nn.init
conv1 = torch.nn.Conv2d(3, 16, 3)
torch.manual_seed(12)
torch.nn.init.normal(conv1.weight, mean=0, std=1)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
print(max(conv1.weight.data.numpy().reshape((-1, 1))))
print(min(conv1.weight.data.numpy().reshape((-1, 1))))
plt.hist(conv1.weight.data.numpy().reshape((-1, 1)), bins=30)
plt.show()

# test conv1.bias
# torch.nn.init.constant(conv1.bias, val=0.1)
