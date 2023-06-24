import torch
from model_mlp import Multilayer

from FashionMINIST.torchtool.check_device import device
from FashionMINIST.torchtool.dataset import test_dataset

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
