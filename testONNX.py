import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision import transforms

ort_session = ort.InferenceSession('baseline.onnx')


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


x = Image.open('test.jpg')
x = x.resize((224, 224))

x = transforms.ToTensor()(x)
x = x.unsqueeze(0)

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

print(ort_outs[0][0].shape)
print(type(ort_outs[0][0][0]))

img_arr = ort_outs[0][0][0]
img_arr = (img_arr * 255).astype(np.uint8)
im = Image.fromarray(img_arr, 'L')
im.save("out.jpeg")
print("Exported model has been tested with ONNXRuntime, and the result looks good!")
