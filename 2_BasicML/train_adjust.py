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
agent.lr_adjust = True
# agent.device = "cpu"

agent.load = False
model_dir = Path("./model")
agent.checkpoint_path = model_dir / "mlp.pth"
png_dir = Path("./png")
agent.fig_path = png_dir / "mlp_adjust.png"

# Initial model class
agent.model_class = Multilayer

# Check path
model_dir.mkdir(exist_ok=True)
png_dir.mkdir(exist_ok=True)

# Start Training
agent.start_training()