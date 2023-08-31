import torch
from model_cnn import LeNet

device = "cuda" if torch.cuda.is_available() else "cpu"
model = LeNet().to(device)
model.load_state_dict(torch.load("model/cnn.pth"))
for name, param in model.named_parameters():
    print(name)
    print(param,end="\n\n")
