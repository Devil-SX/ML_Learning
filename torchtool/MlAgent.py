import time

import matplotlib.pyplot as plt
import torch
from torch import nn

from .check_device import device
from .dataset import test_dataloader, train_dataloader


class MlAgent:
    def __init__(self) -> None:
        self.model_class = None

        self.keep_training : bool = False
        self.if_save : bool = True
        self.epochs = 10
        self.lr = 1e-3
        self.loss_fn = nn.CrossEntropyLoss
        self.optimizer = torch.optim.SGD
        self.schduler = torch.optim.lr_scheduler.LambdaLR
        self.lr_adjust = False

        self.loss_arr = None
        self.time_arr = None
        self.lr_arr = None

        self.checkpoint_path : str = None
        self.fig_path = "loss.png"

        self.device = device
        
        


    def train(self,dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


    def test(self,dataloader, model, loss_fn):
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                loss = loss_fn(pred, y)
                test_loss += loss.item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss


    def start_training(self)->None:
        print(f"using {self.device} device")
        
        model = self.model_class().to(self.device)
        if self.load:
            model.load_state_dict(torch.load(self.checkpoint_path))
            print(f"Load model from {self.checkpoint_path}")


        loss = self.loss_fn()
        optimizer = self.optimizer(model.parameters(), lr=self.lr)
        if self.lr_adjust:
            scheduler = self.schduler(optimizer,lambda epoch: 0.95**epoch)

        self.loss_arr = [0]
        self.time_arr = [0]
        self.lr_arr = [self.lr]
        time_all = 0
        start = time.perf_counter()
        for t in range(self.epochs):
            print(f"Epoch {t+1}/{self.epochs}\n-------------------------------")
            self.train(train_dataloader, model, loss, optimizer)
            self.loss_arr.append(self.test(test_dataloader, model, loss))
            self.time_arr.append(time.perf_counter() - start)
            if self.lr_adjust:
                scheduler.step()
                self.lr_arr.append(scheduler.get_last_lr()[0])
                print(f"lr: {scheduler.get_last_lr()[0]:>7f}")

        print("Done!")
        time_all = time.perf_counter() - start
        print(f"Training time: {time_all:.2f}s")
        self.print()

        if self.if_save:
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f"Saved PyTorch Model State to {self.checkpoint_path}")


    def print(self):
        plt.rc('font',family='Times New Roman')
        ax = plt.gca()
        ax.plot(self.time_arr, self.loss_arr, 'o-', color='blue')
        ax.set_xlabel('time')
        ax.set_ylabel('loss',color='blue')
        if self.lr_adjust:
            plt.title(f"loss-time lr = {self.lr} with schduler")
            ax2 = ax.twinx()
            ax2.plot(self.time_arr, self.lr_arr, 'o-', color='red')
            ax2.set_ylabel('lr',color='red')
        else:
            plt.title(f'loss-time lr = {self.lr}')
        plt.savefig(self.fig_path)
        
