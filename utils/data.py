import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as tf
# from sklearn.model_selection import train_test_split


__all__ = ['dira20']


class dira20(torch.utils.data.Dataset):
    """Some Information about dira20"""

    def __init__(self, root: str, train=True, transform=None):
        super(dira20, self).__init__()

        self.is_train = train
        self.image_path = root
        self.gt_folder = 'gt_image/'
        self.gt_mask_folder = 'gt_binary_image/'
        self.extension = '.png'
        self.transform = transform
        # if self.is_train:
        #     self.image_path = root + 'train/'
        # else:
        #     self.image_path = root + 'images/val/'

        self.list_rgb = [os.path.splitext(os.path.basename(
            x))[0] for x in os.listdir(self.image_path + self.gt_folder)]

        # self.list_depth = self.list_rgb.copy()

        if self.is_train:
            self.list_mask = self.list_rgb.copy()

        self.scale = 1

    def __getitem__(self, index):
        im = Image.open(self.image_path + self.gt_folder +
                        self.list_rgb[index] + self.extension)

        mask = Image.open(self.image_path + self.gt_mask_folder +
                          self.list_mask[index] + self.extension)

        # im.show()
        # mask.show()
        assert im.size == mask.size, \
            f'Image and mask {index} should be the same size, but are {im.size} and {mask.size}'

        # im = self.preprocess(im, self.scale)
        # mask = self.preprocess(mask, self.scale)
        im = tf.ToTensor()(im)
        mask = tf.ToTensor()(mask)
        return im, mask

    def __len__(self):
        return len(self.list_mask)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size

        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'

        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans


if __name__ == "__main__":
    dataset = dira20('/home/ken/Documents/test_tensorRT/dataset/', train=True)
    print(len(dataset))
    for im, label in dataset:
        break
