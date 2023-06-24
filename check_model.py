from pathlib import Path
import onnx
import netron

model_dir = Path('yolov5')
onnx_name = "yolov5n.onnx"

model = onnx.load(str(model_dir/onnx_name))
onnx.checker.check_model(model)
netron.start(str(model_dir/onnx_name))