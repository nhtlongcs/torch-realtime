from torch.utils.data import DataLoader, random_split
from torch import nn
from torchvision.transforms import Compose
from torch.optim import Adam
from utils import *
from models import *


batch_size = 128
lr = 0.001

transform_train = Compose(
    [
    ]
)

transform_val = Compose(
    [
    ]
)

for i in range(2):

    print('===== Run {} ===='.format(i))

    model = Unet()
    optimizer = Adam(model.parameters(), lr=lr)

    train_data = dira20("data/train", train=True)
    val_data = dira20("data/val", train=False)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
