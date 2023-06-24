
import argparse
import torch
from torch import nn
import time
from check_device import device
from utils import train, test
from dataset import train_dataloader, test_dataloader

class CLI:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
        parser.add_argument("--load", action="store_true", help="load the model from last checkpoint")
        self.args = parser.parse_args()


    def start_training(self, model_class, model_name, model_dir):
        model = model_class().to(device)
        if self.args.load:
            model.load_state_dict(torch.load(model_dir / f"{model_name}.pth"))
            print("Load model from last checkpoint.")


        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        # Start Training
        epochs = self.args.epochs
        time_all = 0
        start = time.perf_counter()
        for t in range(epochs):
            print(f"Epoch {t+1}/{epochs}\n-------------------------------")
            train(train_dataloader, model, loss, optimizer)
            test(test_dataloader, model, loss)
        print("Done!")
        time_all = time.perf_counter() - start
        print(f"Training time: {time_all:.2f}s")


        # Save Model
        if not model_dir.is_dir():
            model_dir.mkdir()

        torch.save(model.state_dict(), model_dir / f"{model_name}.pth")
        print(f"Saved PyTorch Model State to {model_name}.pth")
