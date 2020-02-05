from torch import nn
import torch
from torchvision import transforms
# from utils import *
from models import *
from PIL import Image
import time
from torch2trt import torch2trt

weights = torch.load('./baseline.pt')

model = MobileUnet().to('cuda')
model.load_state_dict(weights)
model.eval()


# create example data
dummy_input = torch.ones((1, 3, 224, 224)).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [dummy_input])

img = Image.open('test.jpg')
# img = img.convert('RGB')
img = img.resize((224, 224))
img = transforms.ToTensor()(img).to('cuda')
start = time.time()
for i in range(500):
    with torch.no_grad():
        # inputs = torch.rand((3,224,224)).unsqueeze(0).to('cuda')
        inputs = img.unsqueeze(0).to('cuda')
        # outputs = model(inputs)
        outputs_trt = model_trt(inputs)
        # print(outputs.shape)
end = time.time()
print('fps = {}'.format(500/(end-start)))
transforms.ToPILImage(mode='L')(
    outputs_trt.squeeze(0).cpu()).save('outputs.jpg')
