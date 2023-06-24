from pathlib import Path
from torch import nn
from cli import CLI

model_dir = Path("model")
model_name = "cnn"


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ) # 28*28*1 -> 14*14*6
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ) # 14*14*6 -> 7*7*16
        self.flatten = nn.Flatten()
        self.cnn = nn.Sequential(
            nn.Linear(7*7*16, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        logits = self.cnn(x)
        return logits


if __name__ == "__main__":
    cli = CLI()
    cli.start_training(LeNet, model_name, model_dir)