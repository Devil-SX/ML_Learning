import timeit

import Maxpool2D
import Maxpool2D_m
import numpy as np

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)

    pool1 = Maxpool2D.MaxPoolingLayer(2, 2)
    pool2 = Maxpool2D_m.MaxPoolingLayer(2, 2)

    input = np.random.random([10, 3, 32, 32])


    # accuracy test    
    output1 = pool1.forward(input)
    output2 = pool2.forward(input)
    print(output1.shape, output2.shape)
    print(np.sum(output1 - output2))

    # time test
    time1 = timeit.timeit("pool1.forward(input)", number=100, globals=globals())
    time2 = timeit.timeit("pool2.forward(input)", number=100, globals=globals())

    print("forward time test...")
    print(f"loop style:\t{time1}s")
    print(f"strided style:\t{time2}s")

    # backward test
    top_diff = np.random.random(output1.shape)
    
    time1 = timeit.timeit("pool1.backward(top_diff)", number=100, globals=globals())
    time2 = timeit.timeit("pool2.backward(top_diff)", number=100, globals=globals())

    print("backward time test...")
    print(f"loop style:\t{time1}s")
    print(f"ogrid style:\t{time2}s")
