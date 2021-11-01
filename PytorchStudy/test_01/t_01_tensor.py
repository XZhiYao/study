from __future__ import print_function
import torch
import numpy as np


def t_torch_initial():
    # 没有初始化的5*3矩阵
    x_non_initial = torch.empty(5, 3)
    print(x_non_initial)

    # 随机初始化的5*3矩阵
    x_random_initial = torch.rand(5, 3)
    print(x_random_initial)

    # type : torch long
    x_zero_initial = torch.zeros(5, 3, dtype=torch.long)
    print(x_zero_initial)

    x_data_initial = torch.tensor([5.5, 3, 4.0])
    print(x_data_initial)
    # numpy数组生成张量
    x_numpy_initial = torch.tensor(np.array([5.5, 3, 4.0]))
    print(x_numpy_initial)

    # 新的张量x_data_exist继承已有的张量x_data的结构，指定新的类型
    x_data = torch.ones(5, 3, dtype=torch.double)
    print(x_data)
    x_data_exist = torch.rand_like(x_data, dtype=torch.float)
    print(x_data_exist)

    # 通过指定数据维度生成张量
    shape = (2,3,)
    rand_tensor = torch.rand(shape)
    one_tensor = torch.ones(shape)
    zero_tensor = torch.zeros(shape)
    print(rand_tensor)
    print(one_tensor)
    print(zero_tensor)

    # type : tuple
    print('size:', rand_tensor.size())
    print('size:', x_data_initial.size())
    print('type:', type(x_data_initial.size()))


def t_torch_calculate():
    # 通过索引切片赋值
    tensor = torch.ones(4, 4)
    tensor[:, 1] = 0
    print(tensor)

    # 张量拼接
    tensor_cat_1D = torch.cat([tensor, tensor, tensor], dim=1)
    # tensor_cat_2D = torch.cat([[tensor, tensor], tensor], dim=2)
    print(tensor_cat_1D)
    # print(tensor_cat_2D) --> fail test

    # add 1:
    x = torch.rand(5, 3)
    y = torch.ones(5, 3)
    add_x_y = torch.add(x, y)
    print(x)
    print(y)
    print(add_x_y)

    # add 2:
    add_result = torch.empty(5, 3)
    torch.add(x, y, out=add_result)
    print(add_result)

    # add 3:
    # 任何一个in-place改变张量的操作后面都固定一个_
    y.add_(x)
    print(y)

    print('row:', x[0, :])
    print('column:', x[:, 0])

    # mul做张量位置点乘法运算:
    z = torch.ones(5, 3)
    print(x)
    print(z)
    mul_result_1 = x.mul(z)
    mul_result_2 = x * z
    print(mul_result_1)
    print(mul_result_2)

    # matmul做矩阵乘法运算:
    matmul_result_1 = x.matmul(z.T)
    matmul_result_2 = z @ x.T
    print(matmul_result_1)
    print(matmul_result_2)


def t_torch_resize():
    # view : resize
    x = torch.rand(4, 4)
    y = x.view(16)
    z1 = x.view(-1, 8)
    z2 = x.view(2, 8)
    print(x, x.size())
    print(y, y.size())
    print(z1, z1.size())
    print(z2, z2.size())

    x_single_element = torch.rand(1)
    print(x_single_element)
    print(x_single_element.item())


def t_torch_numpy():
    x = torch.ones(5)
    print(x, type(x))
    y = x.numpy()
    print(y, type(y))

    # x and y share the same memory
    x.add_(2)
    print(x, type(x))
    print(y, type(y))

    a = np.ones(3)
    print(a, type(a))
    b = torch.from_numpy(a)
    print(b, type(b))

    # a and b share the same memory
    np.add(a, 2, out=a)
    print(a, type(a))
    print(b, type(b))

# fail
def t_torch_cuda_device():

    data = torch.rand(5, 3)
    # if GUP enable
    if torch.cuda.is_available():
        # a CUDA device object
        cuda_device = torch.device('cuda')
        # create Tensor object on GPU
        x = torch.ones_like(data, device=cuda_device)
        # data be moved to GUP device
        data = data.to(cuda_device)
        y = x + data
        print(y)
        print(y.to("cpu", torch.double))

if __name__ == '__main__':
    # t_torch_initial()
    # t_torch_calculate()
    t_torch_numpy()
    # t_torch_cuda_device()