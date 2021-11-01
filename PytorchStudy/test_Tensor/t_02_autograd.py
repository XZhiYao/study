import torch

def t_autograd_requires_grad():
    # for train
    x = torch.ones(2, 2, requires_grad=True)
    print(x)

    y = x + 2
    print(y)
    print(y.grad_fn)

    z = y * y * 3
    print(z, z.grad_fn)
    out = z.mean()
    print(out, out.grad_fn)

    a = torch.randn(2, 2)
    a = ((a * 3) / (a - 1))
    print(a.requires_grad)
    a.requires_grad_(True)
    print(a.requires_grad)

    b = (a * a).sum()
    print(b.requires_grad)
    print(b.grad_fn)

def t_autograd_backward():
    x = torch.ones(4, 4,  requires_grad=True)
    print(x.grad)
    y = x + 2
    z = y * y * 3
    out = z.mean()
    out.backward()
    print(x.grad)

def t_autograd_vector():
    x = torch.randn(4, requires_grad=True)
    y = x * 2
    print(y)
    while y.data.norm() < 1000:
        y = y * 2
    print(y)
    print(x.grad)

    # 只要向量积
    vector = torch.tensor([0.1, 1.0, 0.0001, 0.1], dtype=torch.float)
    y.backward(vector)
    print(x.grad)

    print(x.requires_grad)
    print((x ** 2).requires_grad)
    with torch.no_grad():
        print((x ** 2).requires_grad)

if __name__ == '__main__':
    t_autograd_vector()