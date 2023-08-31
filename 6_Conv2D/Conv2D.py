import numpy as np
import time
import timeit


class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride):
        # 卷积层的初始化
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride
        print(
            "\tConvolutional layer with kernel size %d, input channel %d, output channel %d."
            % (self.kernel_size, self.channel_in, self.channel_out)
        )

    def init_param(self, std=0.01):  # 参数初始化
        self.weight = np.random.normal(
            loc=0.0,
            scale=std,
            size=(
                self.channel_in,
                self.kernel_size,
                self.kernel_size,
                self.channel_out,
            ),
        )
        self.bias = np.zeros([self.channel_out])

    def forward(self, input):  # 前向传播的计算
        start_time = time.time()
        self.input = input  # [N, C, H, W]
        # TODO: 边界扩充
        height = input.shape[2] + 2 * self.padding
        width = input.shape[3] + 2 * self.padding
        self.input_pad = np.zeros(
            [self.input.shape[0], self.input.shape[1], height, width], dtype=input.dtype
        )
        self.input_pad[
            ..., self.padding : -self.padding, self.padding : -self.padding
        ] = self.input
        height_out = (height - self.kernel_size) // self.stride + 1
        width_out = (width - self.kernel_size) // self.stride + 1
        # print(height_out, width_out)
        self.output = np.zeros(
            [self.input.shape[0], self.channel_out, height_out, width_out]
        )
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.channel_out):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO: 计算卷积层的前向传播，特征图与卷积核的内积再加偏置
                        self.output[idxn, idxc, idxh, idxw] = (
                            np.sum(
                                self.input_pad[
                                    idxn,
                                    :,
                                    idxh * self.stride : idxh * self.stride
                                    + self.kernel_size,
                                    idxw * self.stride : idxw * self.stride
                                    + self.kernel_size,
                                ]
                                * self.weight[..., idxc],
                                axis=(0, 1, 2),  # For Cin, Kernel_X, Kernel_Y
                            )
                            + self.bias[idxc]
                        )
        return self.output

    def backward(self, top_diff):
        self.d_weight = np.zeros(self.weight.shape)
        self.d_bias = np.zeros(self.bias.shape)
        bottom_diff = np.zeros(self.input_pad.shape)
        for idxn in range(top_diff.shape[0]):  # N
            for idxc in range(top_diff.shape[1]):  # Cout
                for idxh in range(top_diff.shape[2]):  # H
                    for idxw in range(top_diff.shape[3]):  # W
                        # TODO： 计算卷积层的反向传播， 权重、偏置的梯度和本层损失
                        self.d_weight[:, :, :, idxc] += (
                            top_diff[idxn, idxc, idxh, idxw]
                            * self.input_pad[
                                idxn,
                                :,
                                idxh * self.stride : idxh * self.stride
                                + self.kernel_size,
                                idxw * self.stride : idxw * self.stride
                                + self.kernel_size,
                            ]
                        )  # [N, Co, H, W] [N, Ci, H, W] -> [Ci, Kh, Kw, Co]
                        self.d_bias[idxc] += top_diff[idxn, idxc, idxh, idxw]
                        bottom_diff[
                            idxn,
                            :,
                            idxh * self.stride : idxh * self.stride + self.kernel_size,
                            idxw * self.stride : idxw * self.stride + self.kernel_size,
                        ] += (
                            top_diff[idxn, idxc, idxh, idxw]
                            * self.weight[:, :, :, idxc]
                        )
        bottom_diff = bottom_diff[
            :, :, self.padding : -self.padding, self.padding : -self.padding
        ]
        return bottom_diff

    def load_param(self, weight, bias):  # 参数加载
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias


def test_shape():
    conv = ConvolutionalLayer(3, 3, 16, 1, 1)
    conv.init_param()
    weight = np.random.normal(loc=0.0, scale=1.0, size=(3, 3, 3, 16))
    bias = np.random.normal(loc=0.0, scale=1.0, size=(16))
    conv.load_param(weight=weight, bias=bias)
    input = np.random.normal(loc=0.0, scale=1.0, size=(1, 3, 32, 32))
    output = conv.forward(input)
    print(output.shape)


def test_backward():
    print("-" * 10)
    print("Test backward now!")
    conv = ConvolutionalLayer(3, 3, 16, 1, 1)
    conv.init_param()
    weight = np.random.normal(loc=0.0, scale=1.0, size=(3, 3, 3, 16))
    bias = np.random.normal(loc=0.0, scale=1.0, size=(16))
    conv.load_param(weight=weight, bias=bias)
    input = np.random.normal(loc=0.0, scale=1.0, size=(1, 3, 32, 32))
    output = conv.forward(input)
    top_diff = np.random.randn(*output.shape)
    bottom_diff = conv.backward(top_diff=top_diff)

    print("torch is loading ...")
    import torch
    # my weight dim [Cin, Kh, Kw, Cout]
    # torch weight dim [Cout, Cin ,Kh, Kw]
    # input shape in the same

    conv_torch = torch.nn.Conv2d(
        in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
    )
    input_torch = torch.tensor(input, dtype=torch.float32, requires_grad=True)

    with torch.no_grad():
        conv_torch.weight.copy_(torch.tensor(weight.transpose(3, 0, 1, 2))) # my -> torch
        conv_torch.bias.copy_(torch.tensor(bias))

    output_torch = conv_torch(input_torch)
    top_diff_torch = torch.tensor(top_diff, dtype=torch.float32)
    output_torch.backward(top_diff_torch)
    bottom_diff_torch = input_torch.grad 


    diff = np.abs(bottom_diff - bottom_diff_torch.numpy()).sum()
    diff_w = np.abs(conv.d_weight - conv_torch.weight.grad.numpy().transpose(1,2,3,0)).sum() # torch -> my
    diff_b = np.abs(conv.d_bias - conv_torch.bias.grad.numpy()).sum()
    print("Difference between custom and PyTorch's")
    print("bottom_diff:\t", diff)
    print("weight_diff:\t", diff_w)
    print("bias_diff:\t", diff_b)
    assert diff < 1e-5, "Backward implementations are different!"

if __name__ == "__main__":
    # test_shape()
    test_backward()

    # test backward time
    # channel_in = 512
    # channel_out = 512
    # conv = ConvolutionalLayer(kernel_size=3, channel_in=channel_in, channel_out=channel_out, padding=1, stride=1)
    # conv.init_param()
    # weight = np.random.normal(loc=0.0, scale=1.0, size=(channel_in, 3, 3, channel_out))
    # bias = np.random.normal(loc=0.0, scale=1.0, size=(channel_out))
    # conv.load_param(weight=weight, bias=bias)
    # input = np.random.normal(loc=0.0, scale=1.0, size=(1, channel_in, 32, 32))
    # output = conv.forward(input)
    # top_diff = np.random.randn(*output.shape)
    # cmd = "conv.backward(top_diff=top_diff)"
    # print(timeit.timeit(cmd, number=1,globals=globals()))
