import torch
import torch.onnx
from utils import model_select

MODEL_NAME = "EfficientNet"  # EfficientNet, Resnet50 or MobileNet
MODEL_PATH = "results/best_EfficientNet.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model_select(MODEL_NAME)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Exemplo de entrada
x = torch.randn(1, 3, 96, 96)  # Ajuste o tamanho conforme necessário
output_onnx = 'model.onnx'

# Exportar o modelo
torch.onnx.export(model, x, output_onnx, export_params=True, opset_version=10, do_constant_folding=True, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
