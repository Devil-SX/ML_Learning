# FashionMINIST

FashionMINIST 一共有  60,000 个样本

# 使用的模型

- mlp 训练了 25 轮，大约2分半，准确率收敛到 61.1
- mlp 加上BN层后，训练20轮，准确率 87.8
- cnn 训练了 25 轮，大约2分半，准确率收敛到 78.1
- cnn 加上BN层后，训练20轮，准确率 88.9

FashionMNIST 和 MNIST 类似，每张图片物体尺寸和出现位置都一样，所以mlp和cnn差距不大

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