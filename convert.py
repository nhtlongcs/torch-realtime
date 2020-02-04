import torch
from torch2trt import torch2trt
# from torch.utils.data import DataLoader, random_split
from torch import nn
from torchvision import transforms
# from utils import *
from models import *
from PIL import Image
import time
import pdb
from torch2trt import TRTModule

# create some regular pytorch model...
weights = torch.load('./baseline.pt')
model = MobileUnet().to('cuda')
model.load_state_dict(weights)
model.eval()

# create example data
x = torch.ones((1, 3, 224, 224)).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])


# save
# torch.save(model_trt.state_dict(), 'trt_baseline.pth')

# load
# model_trt = TRTModule()
# model_trt.load_state_dict(torch.load('trt_baseline.pth'))

img = Image.open('test.jpg')
# img = img.convert('RGB')
img = img.resize((224,224))
img = transforms.ToTensor()(img).to('cuda')
start = time.time()
for i in range(1):
    with torch.no_grad():
        # inputs = torch.rand((3,224,224)).unsqueeze(0).to('cuda')
        # pdb.set_trace()
        inputs = img.unsqueeze(0).to('cuda')
        outputs = model_trt(inputs)
        print(outputs.shape)
end = time.time()
# print('fps = {}'.format(500/(end-start)))
transforms.ToPILImage(mode='L')(outputs.squeeze(0).cpu()).save('outputs.jpg')