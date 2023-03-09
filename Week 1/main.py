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
import numpy as np
from torchsummary import summary

print("CUDA is available:", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### HYPERPARAMETERS
train_set = 'MIT_small_train_1'
root_dir = '../datasets/' + train_set

train_data_dir= root_dir + '/train'
val_data_dir= root_dir + '/test'
test_data_dir= root_dir + '/test'

img_width = 224
img_height=224
batch_size=4
# epochs = 20
epochs = 10

### CREATE DATASET
# TODO - Put keras transformations
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
        
        self.classifier = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            FireUnit(32, 8, 16, 16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 8)
        )
        
    def forward(self, x):
        x = self.classifier(x)
        return x

net = Net()
net.to(device)

print(summary(net, (3, 256, 256)))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(net.parameters(), lr=0.1)

### TRAIN MODEL
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        

        # print statistics        
        running_loss += loss.item() * inputs.size(0)
        
        # save train accuracy and loss
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_losses.append(running_loss / len(train_dataset))
    train_accuracies.append(100 * correct // total)
    
    running_loss = 0.0
    correct = 0
    total = 0
    # save validation accuracy and loss
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            # calculate outputs by running images through the network
            outputs = net(images)
            
            # TODO - Check val loss is correct            
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        val_accuracies.append(100 * correct // total)
        val_losses.append(running_loss / len(valid_dataset))
        
    print('Epoch: %d, train/val loss: %.3f/%.3f' %(epoch, train_losses[-1], val_losses[-1]))
    
print('Finished Training')

print(f'Accuracy of the network on the test images: {val_accuracies[-1]} %')

offset = 10
plt.figure(figsize=(10,10),dpi=150)
plt.plot(np.arange(0,epochs,offset),train_accuracies[::offset], marker='o', color='orange',label='Train')
plt.plot(np.arange(0,epochs,offset),val_accuracies[::offset], marker='o', color='purple',label='Validation')
plt.grid(color='0.75', linestyle='-', linewidth=0.5)
plt.title('Accuracy',fontsize=25)
plt.xticks(np.arange(0,epochs+offset,offset),fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Accuracy',fontsize=17)
plt.xlabel('Epoch',fontsize=17)
plt.legend(loc='best',fontsize=14)
plt.savefig('./MCV-M5-Team05/Week 1/accuracy.jpg', transparent=False)
plt.close()

# TODO - Check why loss is 0.04, why had we to change the range of ylim to (0,1) instead (0,4)
plt.figure(figsize=(10,10),dpi=150)
plt.plot(np.arange(0,epochs,offset),train_losses[::offset], marker='o', color='orange',label='Train')
plt.plot(np.arange(0,epochs,offset),val_losses[::offset], marker='o', color='purple',label='Validation')
plt.grid(color='0.75', linestyle='-', linewidth=0.5)
plt.title('Loss',fontsize=25)
plt.ylim(0, 4)
plt.xticks(np.arange(0,epochs+offset,offset),fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Loss',fontsize=17)
plt.xlabel('Epoch',fontsize=17)
plt.legend(loc='best',fontsize=14)
plt.savefig('./MCV-M5-Team05/Week 1/loss.jpg', transparent=False)
plt.close()