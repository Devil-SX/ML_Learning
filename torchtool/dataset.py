from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

dataset_dir = Path("../data")

train_dataset = datasets.FashionMNIST(
    root= dataset_dir,
    train=True,
    download=True,
    transform=ToTensor(),
)

test_dataset = datasets.FashionMNIST(
    root=dataset_dir,
    train=False,
    download=True,
    transform=ToTensor(),
)


train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)