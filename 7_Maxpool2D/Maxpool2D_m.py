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

        # output [N, C, H, W]
        # shape [N, C, H, W, K*K]
        shape = (
            *input.shape[:-2],
            height_out,
            width_out,
            self.kernel_size * self.kernel_size,
        )
        strides = (
            *input.strides[:-2],
            input.strides[-2] * self.stride,
            input.strides[-1] * self.stride,
            input.strides[-1],
        )
        self.input_strided = np.lib.stride_tricks.as_strided(
            input, shape=shape, strides=strides
        )
        self.output = np.max(self.input_strided, axis=-1)
        return self.output

    def backward(self, top_diff):
        # print("start maxpool backward")
        # start_time = time.time()
        bottom_diff = np.zeros(self.input.shape)
        # print(f"bottom_diff shape is{bottom_diff.shape}")
        N, C, H, W, _ = self.input_strided.shape
        n, c, h, w = np.ogrid[:N, :C, :H, :W]  # Broadcasting
        max_index = np.argmax(self.input_strided, axis=-1)
        max_mask = np.zeros_like(self.input_strided)
        max_mask[n, c, h, w, max_index] = 1

        # 不重合才能这样操作
        for idxh in range(top_diff.shape[2]):
            for idxw in range(top_diff.shape[3]):
                # print(max_mask[:,:,idxh,idxw].shape)
                # print(top_diff[:, :, idxh, idxw, np.newaxis].shape)
                bottom_diff[
                    :,
                    :,
                    idxh * self.stride : idxh * self.stride + self.kernel_size,
                    idxw * self.stride : idxw * self.stride + self.kernel_size,
                ] = (
                    max_mask[:, :, idxh, idxw] * top_diff[:, :, idxh, idxw, np.newaxis]
                ).reshape(
                    bottom_diff.shape[0],
                    bottom_diff.shape[1],
                    self.kernel_size,
                    self.kernel_size,
                )
        # self.backward_time = time.time() - start_time
        # print("Maxpool backwardtime", self.backward_time)
        return bottom_diff


if __name__ == "__main__":
    pool = MaxPoolingLayer(2, 2)
    output = pool.forward(np.random.random([10, 3, 32, 32]))
    # print(output.shape)
    top_diff = np.random.random(output.shape)
    bottom_diff = pool.backward(top_diff)
