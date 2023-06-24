from pathlib import Path
import argparse
import torch
from train_mlp import Multilayer
from train_cnn import LeNet


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='mlp', choices=["mlp","cnn"],help='mlp or cnn')
args = parser.parse_args()


model_dir = Path('model')
model_name = args.model
if model_name == "mlp":
    model_class = Multilayer
elif model_name == "cnn":
    model_class = LeNet


input_name = f"{model_name}.pth"
output_name = f"{model_name}.onnx"


dummy_input = torch.randn(1, 1, 28, 28)
model = model_class()
model.load_state_dict(torch.load(model_dir/input_name))


torch.onnx.export(model, dummy_input, model_dir/output_name, verbose=True)
