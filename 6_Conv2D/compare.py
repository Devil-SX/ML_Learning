# import Conv2Dexmple
import timeit

import Conv2D
import Conv2D_m
import Conv2Dexmple
import numpy as np

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)

    conv1 = Conv2D_m.ConvolutionalLayer(3, 3, 16, 1, 1)
    conv2 = Conv2D.ConvolutionalLayer(3, 3, 16, 1, 1)
    conv3 = Conv2Dexmple.ConvolutionalLayer(3, 3, 16, 1, 1)
    conv1.init_param()
    conv2.init_param()
    conv3.init_param()
    weight = np.random.normal(loc=0.0, scale=1.0, size=(3, 3, 3, 16))
    bias = np.random.normal(loc=0.0, scale=1.0, size=(16))
    conv1.load_param(weight=weight, bias=bias)
    conv2.load_param(weight=weight, bias=bias)
    conv3.load_param(weight=weight, bias=bias)

    # accuracy test    
    input = np.random.normal(loc=0.0, scale=1.0, size=(1, 3, 32, 32))
    output1 = conv1.forward(input)
    output2 = conv2.forward(input)
    output3 = conv3.forward(input)
    print(output1.shape, output2.shape)
    print("difference between stride style and loop style:")
    print(np.sum(output1 - output2))
    print("difference between stride style and img2col style:")
    print(np.sum(output1 - output3))

    # time test
    time1 = timeit.timeit("conv1.forward(input)", number=100, globals=globals())
    time2 = timeit.timeit("conv2.forward(input)", number=100, globals=globals())
    time3 = timeit.timeit("conv3.forward(input)", number=100, globals=globals())

    print(f"stride style:\t{time1}s")
    print(f"loop style:\t{time2}s")
    print(f"img2col style:\t{time3}s")