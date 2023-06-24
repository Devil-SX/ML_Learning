import argparse
from pathlib import Path

from model_cnn import LeNet
from model_mlp import Multilayer
from torchtool.utils import start_training


# Define CLI interfaces
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
parser.add_argument(
    "--load", action="store_true", help="load the model from last checkpoint"
)
parser.add_argument(
    "--model", type=str, default="mlp", choices=["mlp", "cnn"], help="mlp or cnn"
)
args = parser.parse_args()


# Get parameters
epochs = args.epochs
load = args.load
model_name = args.model
model_dir = Path("./model")
model_path = model_dir / f"{model_name}.pth"


# Check path
if not model_dir.is_dir():
    model_dir.mkdir()


# Initial model class
model_class = (Multilayer
         if model_name == "mlp"
         else LeNet
         if model_name == "cnn"
         else None)


# Start Training
start_training(model_class,model_path,load,epochs)