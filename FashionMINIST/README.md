# FashionMINIST

FashionMINIST 一共有  60,000 个样本

# 使用的模型

- mlp 训练了 25 轮，大约2分半，准确率收敛到 61.1
- cnn 训练了 25 轮，大约2分半，准确率收敛到 78.1

# Get Started

```shell
python ./cli.py --help
```

```
usage: cli.py [-h] [--epochs EPOCHS] [--load] [--model {mlp,cnn}]

options:
  -h, --help         show this help message and exit
  --epochs EPOCHS    number of epochs
  --load             load the model from last checkpoint
  --model {mlp,cnn}  mlp or cnn
```

默认保存/加载模型路径在 `./model` 下

[更多benchmark](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/#)