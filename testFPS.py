# from torch.utils.data import DataLoader, random_split
from torch import nn
import torch
from torchvision import transforms
# from utils import *
from models import *
from PIL import Image
import time
weights = torch.load('./baseline.pt')

model = MobileUnet().to('cuda')
model.load_state_dict(weights)
model.eval()

img = Image.open('test.jpg')
# img = img.convert('RGB')
img = img.resize((224, 224))
img = transforms.ToTensor()(img).to('cuda')
start = time.time()
for i in range(500):
    with torch.no_grad():
        # inputs = torch.rand((3,224,224)).unsqueeze(0).to('cuda')
        inputs = img.unsqueeze(0).to('cuda')
        outputs = model(inputs)
        # print(outputs.shape)
end = time.time()
print('fps = {}'.format(500/(end-start)))
transforms.ToPILImage(mode='L')(outputs.squeeze(0).cpu()).save('outputs.jpg')
