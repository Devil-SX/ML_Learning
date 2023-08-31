# Experiments

- Layer Shape  (2,2)
- Input Shape  (10, 3, 32, 32)
- Forward 100 Times

| Algorithm | Consumption(s) |
|---|---|
| Loop | 2.010 |
| as_strided + tensordot | 0.064 |

