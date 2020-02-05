from torch import nn
import torch
from torchvision import transforms
# from utils import *
from models import *
from PIL import Image
import time
weights = torch.load('./baseline.pth')
model = MobileUnet().to('cuda')
model.load_state_dict(weights)

model.eval()  # important
input_names = ["input"]
output_names = ["output"]
with torch.no_grad():
    dummy_input = torch.autograd.Variable(
        torch.rand(1, 3, 224, 224, requires_grad=True).cuda())
    torch.onnx.export(model, dummy_input, 'baseline.onnx', verbose=True,
                      do_constant_folding=True, export_params=True,
                      input_names=input_names, output_names=output_names)
print('Done.')
