from train_mlp import Multilayer
import torch
from dataset import test_dataset
from check_device import device


model = Multilayer().to(device)
model.load_state_dict(torch.load("model/mlp.pth"))
print(" Load Successful!")

model.eval()

with torch.no_grad():
  x = test_dataset[25][0].to(device)
  y = model(x)
  print(f"Prediction Result \t {y}")
  print(f"Class Result\t {y[0].argmax()}")
  print(f"Ideal \t {test_dataset[25][1]}")
