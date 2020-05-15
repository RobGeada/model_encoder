import numpy as np
import os

import torch
from torchvision import datasets, transforms


# === CUTOUT ===========================================================================================================
# adapted from github.com/hysts/pytorch_cutout
class Cutout:
    def __init__(self, mask_size, p, mask_color=(0, 0, 0)):
        self.mask_size = mask_size
        self.p = p
        self.mask_color = mask_color
        self.mask_size_half = mask_size // 2
        self.offset = 1 if mask_size % 2 == 0 else 0

    def __call__(self, image):
        image = np.asarray(image).copy()

        if np.random.random() > self.p:
            return image

        h, w = image.shape[:2]

        cxmin, cxmax = 0, w + self.offset
        cymin, cymax = 0, h + self.offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - self.mask_size_half
        ymin = cy - self.mask_size_half
        xmax = xmin + self.mask_size
        ymax = ymin + self.mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = self.mask_color
        return image


# === DATA HELPERS =====================================================================================================
# Load the CIFAR10 dataset into train and test dataloaders
def load_cifar(batch_size):
    data_path = os.getcwd() + "/data/"
    download = 'CIFAR10' not in os.listdir(data_path)

    # dataset configuration
    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]

    train_data = datasets.CIFAR10(data_path + "CIFAR10",
                                  train=True,
                                  download=download,
                                  transform=transforms.Compose([
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      Cutout(mask_size=16, p=.5, mask_color=MEAN),
                                      transforms.ToTensor(),
                                      transforms.Normalize(MEAN, STD)]
                                  ))
    test_data = datasets.CIFAR10(data_path + "CIFAR10",
                                 train=False,
                                 download=download,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(MEAN, STD)]
                                 ))

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=False)

    # get/load dataset metadata
    for img, target in train_loader:
        data_shape = img.shape
        break

    return (train_loader, test_loader), data_shape

