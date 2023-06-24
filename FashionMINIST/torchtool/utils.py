import time

import torch
from torch import nn

from .check_device import device
from .dataset import test_dataloader, train_dataloader


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def start_training(model_class, model_path,load:bool, epochs:int)->None:
    print(f"using {device} device")
    model = model_class().to(device)
    if load:
        model.load_state_dict(torch.load(model_path))
        print(f"Load model from {model_path}")


    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Start Training
    epochs = epochs
    time_all = 0
    start = time.perf_counter()
    for t in range(epochs):
        print(f"Epoch {t+1}/{epochs}\n-------------------------------")
        train(train_dataloader, model, loss, optimizer)
        test(test_dataloader, model, loss)
    print("Done!")
    time_all = time.perf_counter() - start
    print(f"Training time: {time_all:.2f}s")


    torch.save(model.state_dict(), model_path)
    print(f"Saved PyTorch Model State to {model_path}")