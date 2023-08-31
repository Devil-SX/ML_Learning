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
        self.height_out = (height - self.kernel_size) // self.stride + 1
        self.width_out = (width - self.kernel_size) // self.stride + 1

        self.stride_shape = (
            self.input_pad.shape[0],
            self.input_pad.shape[1],
            self.height_out,
            self.width_out,
            self.kernel_size,
            self.kernel_size,
        )
        self.stride_strides = (
            *self.input_pad.strides[:-2],  # for N and C
            self.input_pad.strides[-2] * self.stride,  # for H out
            self.input_pad.strides[-1] * self.stride,  # for W out
            *self.input_pad.strides[-2:],  # for kernel size
        )
        input_strides = np.lib.stride_tricks.as_strided(
            self.input_pad, shape=self.stride_shape, strides=self.stride_strides
        )

        self.output = (
            np.einsum("nchwxy,cxyo->nohw", input_strides, self.weight, optimize=True)
            + self.bias[np.newaxis, :, np.newaxis, np.newaxis]
        )  # [N,Cout,H,W] + [1,Cout,1,1]

        # self.output = np.tensordot(input_strides, self.weight, axes=[(1,4,5),(0,1,2)]) + self.bias
        # self.output = np.transpose(self.output, (0, 3, 1, 2))
        return self.output

    def backward(self, top_diff):
        # top_diff [N, Cout, H, W]
        input_strides = np.lib.stride_tricks.as_strided(
            self.input_pad, shape=self.stride_shape, strides=self.stride_strides
        )

        self.d_weight = np.tensordot(input_strides, top_diff,axes=[(0,2,3),(0,2,3)])
        self.d_bias = np.sum(top_diff, axis=(0, 2, 3))

        pad_height = self.input_pad.shape[2]
        pad_width = self.input_pad.shape[3]
        pad_h = pad_height - self.kernel_size
        pad_w = pad_width - self.kernel_size
        pad_filt = np.pad(
            self.weight, ((0, 0),  (pad_h, pad_h), (pad_w, pad_w), (0, 0)), "constant"
        )
        sub_windows = np.lib.stride_tricks.as_strided(
            pad_filt,
            shape=(
                self.channel_in,
                self.height_out,
                self.width_out,
                pad_height,
                pad_width,
                self.channel_out,
            ),
            strides=(
                pad_filt.strides[0], # C
                pad_filt.strides[1]* self.stride, # A
                pad_filt.strides[2]* self.stride, # B
                pad_filt.strides[1], # H
                pad_filt.strides[2], # W
                pad_filt.strides[3], # O
            ),
        )

        bottom_diff = np.einsum("cabhwo,noab->nchw",sub_windows, top_diff[:,:,::-1,::-1], optimize=True)

        bottom_diff = bottom_diff[
            :, :, self.padding : -self.padding, self.padding : -self.padding
        ]
        return bottom_diff

    def load_param(self, weight, bias):  # 参数加载
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias


def shape_test():
    conv = ConvolutionalLayer(3, 3, 16, 1, 1)
    conv.init_param()
    weight = np.random.normal(loc=0.0, scale=1.0, size=(3, 3, 3, 16))
    bias = np.random.normal(loc=0.0, scale=1.0, size=(16))
    conv.load_param(weight=weight, bias=bias)
    input = np.random.normal(loc=0.0, scale=1.0, size=(1, 3, 32, 32))
    output = conv.forward(input)
    print(output.shape)


def stride_test():
    conv = ConvolutionalLayer(
        kernel_size=2, channel_in=1, channel_out=1, padding=1, stride=1
    )
    conv.init_param()
    weight = np.random.normal(loc=0.0, scale=1.0, size=(1, 2, 2, 1))
    bias = np.random.normal(loc=0.0, scale=1.0, size=(1))
    conv.load_param(weight=weight, bias=bias)
    input = np.arange(9).reshape(1, 1, 3, 3)
    # print(input)
    output = conv.forward(input)
    # print(output.shape)


def test_backward_accuracy():
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
        conv_torch.weight.copy_(
            torch.tensor(weight.transpose(3, 0, 1, 2))
        )  # my -> torch
        conv_torch.bias.copy_(torch.tensor(bias))

    output_torch = conv_torch(input_torch)
    top_diff_torch = torch.tensor(top_diff, dtype=torch.float32)
    output_torch.backward(top_diff_torch)
    bottom_diff_torch = input_torch.grad

    diff = np.abs(bottom_diff - bottom_diff_torch.numpy()).sum()
    diff_w = np.abs(
        conv.d_weight - conv_torch.weight.grad.numpy().transpose(1, 2, 3, 0)
    ).sum()  # torch -> my
    diff_b = np.abs(conv.d_bias - conv_torch.bias.grad.numpy()).sum()
    print("Difference between custom and PyTorch's")
    print("bottom_diff:\t", diff)
    print("weight_diff:\t", diff_w)
    print("bias_diff:\t", diff_b)
    assert diff < 1e-4, "Backward implementations are different!"


if __name__ == "__main__":
    # shape_test()
    stride_test()
    # test_backward_accuracy()

    # test backward time
    # channel_in = 512
    # channel_out = 512
    # conv = ConvolutionalLayer(
    #     kernel_size=3,
    #     channel_in=channel_in,
    #     channel_out=channel_out,
    #     padding=1,
    #     stride=1,
    # )
    # conv.init_param()
    # weight = np.random.normal(loc=0.0, scale=1.0, size=(channel_in, 3, 3, channel_out))
    # bias = np.random.normal(loc=0.0, scale=1.0, size=(channel_out))
    # conv.load_param(weight=weight, bias=bias)
    # input = np.random.normal(loc=0.0, scale=1.0, size=(1, channel_in, 32, 32))
    # output = conv.forward(input)
    # top_diff = np.random.randn(*output.shape)
    # cmd = "conv.backward(top_diff=top_diff)"
    # print(timeit.timeit(cmd, number=1, globals=globals()))
