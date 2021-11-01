import torch
# resnet config
depth = 50
res5_dilation = 1
in_channels = 64 # stage 1 input; stage 0 output is 64 * 64
out_channels = 256
norm = "FrozenBN"
num_groups = 1
width_per_group = 64
bottleneck_channels = num_groups * width_per_group
stride_in_1x1 = True
deform_on_per_stage = [False, False, False, False]

num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]

class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

class BasicBlock(torch.nn.Module):


    def __init__(self, in_channels, out_channels, *, stride=1, norm=torch.nn.BatchNorm2d):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                # norm=get_norm(norm, out_channels),
                norm=norm
            )
        else:
            self.shortcut = None

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            # norm=get_norm(norm, out_channels),
            norm=norm
        )

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            # norm=get_norm(norm, out_channels),
            norm=norm
        )

class BottleneckBlock(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,

            bottleneck_channels,
            stride=1,
            num_groups=1,
            norm=torch.nn.BatchNorm2d,
            stride_in_1x1=False,
            dilation=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                # norm=get_norm(norm, out_channels),
                norm=norm
            )
        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        print('stride: ', stride_1x1, stride_3x3)


if __name__ == '__main__':
    print(depth)
    print(num_blocks_per_stage)
    basicBlock = BasicBlock(in_channels, out_channels)
    print('bottleneck_channels:', bottleneck_channels)
    bottleneckBlock = BottleneckBlock(in_channels, out_channels, bottleneck_channels)
    for idx, stage_idx in enumerate(range(2, 6)):
        print('loop:', idx, stage_idx)
        dilation = res5_dilation if stage_idx == 5 else 1
        print(res5_dilation, dilation)
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        print(first_stride)
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "stride_per_block": [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
        }
        print(stage_kargs)
        if depth in [18, 34]:
            stage_kargs["block_class"] = BasicBlock
        else:
            stage_kargs["bottleneck_channels"] = bottleneck_channels
            stage_kargs["stride_in_1x1"] = stride_in_1x1
            stage_kargs["dilation"] = dilation
            stage_kargs["num_groups"] = num_groups
            if deform_on_per_stage[idx]:
                stage_kargs["block_class"] = None
                stage_kargs["deform_modulated"] = None
                stage_kargs["deform_num_groups"] = None
            else:
                stage_kargs["block_class"] = BottleneckBlock
        print(stage_kargs)

def f(a,b):
    print('输出:',a,b)

f(**{"a":1,"b":2})
f(a=1,b=2)