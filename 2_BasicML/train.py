#!python
from pathlib import Path
from model_cnn import LeNet
from model_mlp import Multilayer

import sys
sys.path.append('../')
from torchtool import MlAgent


agent = MlAgent()

# Get parameters
agent.epochs = 10
agent.lr = 5e-3
# agent.device = "cpu"

agent.load = False
model_dir = Path("./model")
agent.checkpoint_path = model_dir / "mlp.pth"
png_dir = Path("./png")
agent.fig_path = png_dir / "mlp.png"

# Initial model class
agent.model_class = Multilayer

# Check path
if not model_dir.is_dir():
    model_dir.mkdir()

# Start Training
agent.start_training()