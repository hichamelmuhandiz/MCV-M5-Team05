import torch
import torchvision.transforms as transform
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import torchvision
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import sys

print("Cuda is available:", torch.cuda.is_available())

### HYPERPARAMETERS
train_set = 'MIT_small_train_1'
root_dir = '../datasets/' + train_set

train_data_dir= root_dir + '/train'
val_data_dir= root_dir + '/test'
test_data_dir= root_dir + '/test'

img_width = 224
img_height=224
batch_size=4
epochs = 1000

### CREATE DATASET
transformation = transform.Compose([
    # you can add other transformations in this list
    ToTensor()
])

train_dataset = torchvision.datasets.ImageFolder(root="MCV-M5-Team05/datasets/MIT_small_train_1/train", transform=transformation)
valid_dataset = torchvision.datasets.ImageFolder(root="MCV-M5-Team05/datasets/MIT_small_train_1/test", transform=transformation)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

## Show a sample of our training set
train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
img = torch.permute(img, (1,2,0))
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

