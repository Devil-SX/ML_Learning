import numpy as np
import time


class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride):  # 最大池化层的初始化
        self.kernel_size = kernel_size
        self.stride = stride
        print(
            "\tMax pooling layer with kernel size %d, stride %d."
            % (self.kernel_size, self.stride)
        )

    def forward(self, input):  # 前向传播的计算
        start_time = time.time()
        self.input = input  # [N, C, H, W]
        self.max_index = np.zeros(self.input.shape)
        height = self.input.shape[2]
        width = self.input.shape[3]
        height_out = (height - self.kernel_size) // self.stride + 1
        width_out = (width - self.kernel_size) // self.stride + 1
        self.output = np.zeros(
            [self.input.shape[0], self.input.shape[1], height_out, width_out]
        )
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.input.shape[1]):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO： 计算最大池化层的前向传播， 取池化窗口内的最大值
                        self.output[idxn, idxc, idxh, idxw] = np.max(
                            self.input[
                                idxn,
                                idxc,
                                idxh * self.stride : idxh * self.stride
                                + self.kernel_size,
                                idxw * self.stride : idxw * self.stride
                                + self.kernel_size,
                            ]
                        )
        return self.output

    def backward(self, top_diff):
        # print("start maxpool backward")
        # start_time = time.time()
        bottom_diff = np.zeros(self.input.shape)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        # TODO: 最大池化层的反向传播， 计算池化窗口中最大值位置， 并传递损失
                        max_index = np.argmax(
                            self.input[
                                idxn,
                                idxc,
                                idxh * self.stride : idxh * self.stride
                                + self.kernel_size,
                                idxw * self.stride : idxw * self.stride
                                + self.kernel_size,
                            ]
                        )
                        max_index = np.unravel_index(
                            max_index, (self.kernel_size, self.kernel_size)
                        )
                        bottom_diff[
                            idxn,
                            idxc,
                            idxh * self.stride + max_index[0],
                            idxw * self.stride + max_index[1],
                        ] = top_diff[idxn, idxc, idxh, idxw]
        # self.backward_time = time.time() - start_time
        # print("Maxpool backwardtime", self.backward_time)
        return bottom_diff


if __name__ == "__main__":
    pool = MaxPoolingLayer(2, 2)
    output = pool.forward(np.random.random([10, 3, 32, 32]))
    print(output.shape)
