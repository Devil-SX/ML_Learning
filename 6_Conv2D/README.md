3 effcient ways of implementing Conv2D

- Loop ways
- img2col
- use `lib.as_strided` + `tensordot` or `einsum`


# Experiments

## Forward

- Layer Shape  (3, 3, 16, 1, 1)
- Input Shape  (1, 3, 32, 32)
- Forward 100 Times


| Algorithm | Consumption(s) |
|---|---|
| Loop | 6.122 |
| img2col | 0.1444 |
| as_strided + einsum | 0.068 |
| as_strided + einsum + optimize | 0.0118 |
| as_strided + tensordot | 0.0113 |


## Backward

- Layer Shape  (3, 512, 512, 1, 1)
- Input Shape  (1, 512, 32, 32)
- Backward 1 Times


| Algorithm | Consumption(s) |
|---|---|
| 4-Loop(N,C,H,W) | 71.7204 |
| 2-Loop(H,W) + einsum + optimize | 4.4878 |
| 2-Loop(H,W) + tensordot | 1.9864 |
| img2col | 0.0977 |

img2col 虽然也是2层循环，但 Loop 内部没有乘法，故img2col速度更快