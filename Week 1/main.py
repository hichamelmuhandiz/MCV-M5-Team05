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
import torch.optim as optim

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
epochs = 2
# epochs = 200

### CREATE DATASET
transformation = transform.Compose([
    # you can add other transformations in this list
    ToTensor()
])

train_dataset = torchvision.datasets.ImageFolder(root="MCV-M5-Team05/datasets/MIT_small_train_1/train", transform=transformation)
valid_dataset = torchvision.datasets.ImageFolder(root="MCV-M5-Team05/datasets/MIT_small_train_1/test", transform=transformation)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)


print('Classes: ', train_dataset.classes)

## Show a sample of our training set
train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
img = torch.permute(img, (1,2,0))
label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
print(f"Label: {train_dataset.classes[label]}")

### CREATE MODEL
class FireUnit(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )
        
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            FireUnit(32, 8, 16, 16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 1)),
            nn.Linear(1024, 8)
        )
        
    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        x = self.classifier(x)
        return x

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

### TRAIN MODEL
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 49 == 48:    # print every 49 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

### TEST MODEL
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in valid_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')