import torch
import torch.nn as nn
import logging
from unet import MobileUnet
from PIL import Image
import torchvision.transforms as transforms



if __name__ == "__main__":
    # rand_input = torch.rand((1, 3, 224, 224))
    rand_input = Image.open('./test.jpg')

    transform_val = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),
        ]
    )
    rand_input = transform_val(rand_input).unsqueeze(0)

    logging.basicConfig()

    model = MobileUnet()
    model.load_state_dict(torch.load('./baseline.pth', map_location='cpu'))
    model.eval()

    script_model = torch.jit.trace(model, rand_input)
    print(script_model.code)

    with torch.no_grad():
        output = script_model(rand_input)
    output = torch.argmax(output, dim = 1).float()

    output = transforms.ToPILImage(mode='L')(output.squeeze(0).cpu())
    output = output.resize((320, 240))
    output.save('output_2.jpg')

    script_model.save('./baseline_jit.pth')

    
