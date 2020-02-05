import onnx
import onnx_tensorrt.backend as backend
import numpy as np
import cv2

model = onnx.load("./baseline.onnx")
onnx.checker.check_model(model)
# print(onnx.helper.printable_graph(model.graph))

engine = backend.prepare(model, device='CUDA', max_batch_size=1)

input_data = cv2.imread('test.jpg', cv2.IMREAD_COLOR).astype(np.float32)
output_data = engine.run(input_data)[0]
# print(output_data)
print(output_data.shape)

cv2.imshow(output_data)
