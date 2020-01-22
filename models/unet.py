from torch import nn

__all__ = ['Unet']


class Unet(nn.Module):
    """Some Information about Unet"""

    def __init__(self):
        super(Unet, self).__init__()

    def forward(self, x):

        return x
