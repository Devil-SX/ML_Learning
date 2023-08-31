import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd


class BatchNormLayer:
    def __init__(self, eps=1e-5):
        self.eps = eps
        print("\tBN layer")

    def forward(self, input):
        self.input = input
        self.batch_size = self.input.shape[0]
        print("\tbatch size: ", self.batch_size)
        self.var = np.var(self.input, axis=0, keepdims=True)
        self.nu = np.mean(self.input, axis=0, keepdims=True)
        self.output = (self.input - self.nu) / np.sqrt(self.var + self.eps)
        return self.output

    def backward(self, top_diff):
        var_d = (
            -1
            / 2.0
            * np.sum((self.input - self.nu) * top_diff, axis=0, keepdims=True)
            * np.power(self.var + self.eps, -1.5)
        )
        nu_d = var_d * (
            -2 / self.batch_size * np.sum(self.input - self.nu, axis=0, keepdims=True)
        ) - np.sum(top_diff, axis=0, keepdims=True) / np.sqrt(self.var + self.eps)
        bottom_diff = (
            nu_d / self.batch_size
            + var_d * (2 / self.batch_size * (self.input - self.nu))
            + top_diff / np.sqrt(self.var + self.eps)
        )
        return bottom_diff


# Test
# Random input
input_np = np.random.randn(32, 64)  # 示例：batch size为32，feature数为64
input_torch = torch.tensor(input_np, dtype=torch.float32, requires_grad=True) 
# Random backprop gradient
top_diff_np = np.random.randn(32, 64)
top_diff_torch = torch.tensor(top_diff_np, dtype=torch.float32)

# Forward
# Numpy
np_bn = BatchNormLayer()
x_hat_np = np_bn.forward(input_np)
# Pytorch
torch_bn = nn.BatchNorm1d(
    64, eps=1e-5, momentum=0, affine=False
)  # affine=False表示没有偏移和缩放
# torch_bn.eval()
x_hat_torch = torch_bn(input_torch)

# Backward
x_d_np = np_bn.backward(top_diff_np)

x_hat_torch.backward(top_diff_torch)
x_d_torch = input_torch.grad

# 比较两个输出
print("Your BN forward output: ", np.sum(np.abs(x_hat_np - x_hat_torch.data.numpy())))
# 比较两个梯度
print("Your BN backward diff: ", np.sum(np.abs(x_d_np - x_d_torch.data.numpy())))
