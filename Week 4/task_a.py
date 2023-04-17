from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import ToTensor
from torchvision.transforms import transforms as transform
from torchsummary import summary
import torchvision.models as models
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import PrecisionRecallDisplay, accuracy_score, precision_recall_curve
import matplotlib.pyplot as plt
import sys
import os
from PIL import Image
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.neighbors import KNeighborsClassifier

# create a SummaryWriter object
# writer = SummaryWriter()

print("CUDA is available:", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### HYPERPARAMETERS
root_dir = '/home/mcv/datasets/MIT_split'
train_data_dir= root_dir + '/train'
val_data_dir= root_dir + '/test'
test_data_dir= root_dir + '/test'

img_width = 224
img_height = 224
batch_size = 64
epochs = 30


### CREATE DATASET
transformation_train = transform.Compose([
    transform.Resize((224, 224)),
    transform.ToTensor()
])
# No need to use data augmentation in validation
transformation_val = transform.Compose([
    transform.Resize((224, 224)),
    transform.ToTensor()
])

train_dataset = torchvision.datasets.ImageFolder(root=train_data_dir, transform=transformation_train)
valid_dataset = torchvision.datasets.ImageFolder(root=test_data_dir,  transform=transformation_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

print('Classes: ', train_dataset.classes)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Linear(resnet.fc.in_features, 8)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

net = Net()
net.to(device)

criterion = nn.CrossEntropyLoss()

# optimizer = optim.Adadelta(net.parameters(), lr=0.1)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

softmax = torch.nn.Softmax(dim=1)

### TRAIN MODEL
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []
y_pred = []

for epoch in range(epochs):  # loop over the dataset multiple times
    print ('epoch:', epoch)
    running_loss = 0.0
    correct = 0
    total = 0
    net.train() # Set the model to training mode
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
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Save metrics for training set
    train_accuracy = 100 * correct / total
    train_loss = running_loss / len(train_loader)
    train_accuracies.append(train_accuracy)
    train_losses.append(train_loss)

    print('accuracy:', train_accuracy)
    with torch.no_grad(): 
        net.eval() 
        running_loss = 0.0
        correct = 0
        total = 0
        for data in valid_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            running_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            outputs = softmax(outputs)
            if epoch == epochs-1:
                y_pred.extend(np.max(outputs.detach().cpu().numpy(), axis=1).flatten().tolist())

        # Save metrics for validation set
        val_accuracy = 100 * correct / total
        val_loss = running_loss / len(valid_loader)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)
        print('val_accuracy', val_accuracy)

# Plot Loss
fig, ax = plt.subplots(1,1,figsize=(10,10), dpi=200)
ax.plot(np.arange(0,epochs,2),train_losses[::2],label='Train Loss',marker='o', color='orange')
ax.plot(np.arange(0,epochs,2),val_losses[::2],label='Val Loss',marker='o', color='purple')
plt.grid(color='0.75', linestyle='-', linewidth=0.5)
plt.title('Loss',fontsize=25)
plt.ylabel('Loss',fontsize=17)
plt.xlabel('Epoch',fontsize=17)
plt.legend(loc='best',fontsize=14)
plt.savefig('loss_a.png')
plt.close()

# Remove the last layer (the classification layer)
net_retrieval = torch.nn.Sequential(*(list(net.children())[:-1]))

# Evaluate on validation set
with torch.no_grad(): 
    net_retrieval.eval() 
    running_loss = 0.0
    correct = 0
    total = 0
    val_features = np.zeros((0, 2048))
    val_images = np.zeros((0, 3, img_height, img_width))
    val_targets = []
    for data in valid_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net_retrieval(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        n_samples = outputs.shape[0]
        outputs_flat = outputs.reshape(n_samples, outputs.shape[1])
        val_features = np.concatenate((val_features, outputs_flat.detach().cpu().numpy()), axis=0)
        val_targets.extend(labels.detach().cpu().numpy().flatten().tolist())
        val_images = np.concatenate((val_images, images.detach().cpu().numpy()), axis=0)

val_features = np.array(val_features)
val_targets = np.array(val_targets)
print(val_features.shape,val_targets.shape)

# Evaluate on training set
with torch.no_grad(): 
    net_retrieval.eval() 
    running_loss = 0.0
    correct = 0
    total = 0
    train_features = np.zeros((0, 2048))
    train_images = np.zeros((0, 3, img_height, img_width))
    train_targets = []
    for data in train_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net_retrieval(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        n_samples = outputs.shape[0]
        outputs_flat = outputs.reshape(n_samples, outputs.shape[1])
        train_features = np.concatenate((train_features, outputs_flat.detach().cpu().numpy()), axis=0)
        train_targets.extend(labels.detach().cpu().numpy().flatten().tolist())
        train_images = np.concatenate((train_images, images.detach().cpu().numpy()), axis=0)

train_features = np.array(train_features)
train_targets = np.array(train_targets)
print(train_features.shape,train_targets.shape)

class_names = train_dataset.classes

k = 10 # number of neighbors
#knn = NearestNeighbors(n_neighbors=k)
#knn.fit(train_features)
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(train_features,train_targets)
distances, indices = knn.kneighbors(val_features)
print('Retrieved images:', indices)

# Plot retrieval
fig, ax = plt.subplots(4, 6, figsize=(10, 10), dpi=200)
for i in range (0,4):
    ax[i][0].imshow(np.moveaxis(val_images[i], 0, -1))
    ax[i][0].set_xticks([])
    ax[i][0].set_yticks([])
    class_name = class_names[val_targets[i]]
    ax[i][0].set_title("Test img:\n" + class_name, fontsize=9)
    for j in range (0,5):
        ax[i][j+1].imshow(np.moveaxis(train_images[indices[i,j]], 0, -1))
        ax[i][j+1].set_xticks([])
        ax[i][j+1].set_yticks([])
        class_name = class_names[train_targets[indices[i][j]]]
        ax[i][j+1].set_title("Retrieved img:\n" + class_name, fontsize=9)
fig.tight_layout()
plt.savefig('retrieval_a.png')
plt.close()


# Compute Precision Recall curve for Image Classification
y_pred = np.array(y_pred).flatten()

fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=200)
ax.set_title("Precision-Recall curve", size=16)
for class_id, class_name in enumerate(class_names):
    PrecisionRecallDisplay.from_predictions(
        np.where(val_targets == class_id, 1, 0),
        np.where(val_targets == class_id, y_pred, 1 - y_pred),
        ax=ax,name=class_name)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig("pr_curve_classi_a.png")


# Compute Precision Recall curve for Image Retrieval
val_prob = knn.predict_proba(val_features)
fig, ax = plt.subplots(1, 1, figsize=(10,10), dpi=150)
ax.set_title("Precision-Recall curve", size=16)
for class_id, class_name in enumerate(class_names):
    PrecisionRecallDisplay.from_predictions(np.where(val_targets==class_id, 1, 0),
                                            val_prob[:, class_id],
                                            ax=ax, name=class_name)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.savefig("pr_curve_a_cool.png")


## Compute Precision Recall curve for image retrieval using knn = NearestNeighbors(n_neighbors=k)
# fig, ax = plt.subplots(1, 1, figsize=(10,10), dpi=200)
# ax.set_title("Precision-Recall curve", size=16)
# for class_id, class_name in enumerate(class_names):
#     PrecisionRecallDisplay.from_predictions(np.where(val_targets==class_id, 1, 0),
#                                             np.where(train_targets[indices][:, 0]==class_id, 1, 0),
#                                             ax=ax, name=class_name)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.legend(loc='lower left')
# plt.savefig('pr_curve_a.png')


# Compute the evaluation metrics
APs = []
Precisions_at_1 = []
Precisions_at_5 = []

# Compute MAP
# Convert integer targets to binary targets
binary_val_targets = np.zeros((val_features.shape[0], 8))
binary_val_targets[np.arange(val_features.shape[0]), val_targets] = 1

binary_train_targets = np.zeros((val_features.shape[0], 8))
binary_train_targets[np.arange(val_features.shape[0])[:,None], train_targets[indices]] = 1

# Compute average precision 
for i in range(val_features.shape[0]):
    AP = average_precision_score(binary_val_targets[i], binary_train_targets[i])
    APs.append(AP)

MAP = np.mean(APs)

for i, (dists, idxs, target) in enumerate(zip(distances, indices, val_targets)):
    # Compute the precision at 1
    if train_targets[idxs[0]] == target:
        Precisions_at_1.append(1.0)
    else:
        Precisions_at_1.append(0.0)

    # Compute the precision at 5
    Precisions_at_5 = []
    for idx, target in zip(indices, val_targets):
        hits = np.isin(train_targets[idx][:5], target)
        precision_at_5 = np.sum(hits) / 5.0
        Precisions_at_5.append(precision_at_5)

# Compute the precision at 1 and precision at 5
Prec_1 = np.mean(Precisions_at_1)
Prec_5 = np.mean(Precisions_at_5)
print('MAP:', MAP)
print('Prec@1:', Prec_1)
print('Prec@5:', Prec_5)


### TASK D

def plot_embeddings(embeddings, targets, title='', xlim=None, ylim=None):
    num_classes = ['Opencountry', 'coast', 'forest', 'highway', 'inside_city', 'mountain', 'street', 'tallbuilding']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    plt.figure(figsize=(10,10))
    plt.title(title)
    for i in range(8):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(num_classes)
    plt.savefig(title + '_a.png')

# Train
# Compute PCA
pca = PCA(n_components=2)
train_embeddings_pca = pca.fit_transform(train_features)
plot_embeddings(train_embeddings_pca, train_targets, title='Embeddings train with PCA')
# Compute TSNE
train_embeddings_tsne = TSNE(n_components=2).fit_transform(train_features)
plot_embeddings(train_embeddings_tsne, train_targets, title='Embeddings train with TSNE')
# Compute UMAP
train_embeddings_umap = umap.UMAP().fit_transform(train_features)
plot_embeddings(train_embeddings_umap, train_targets, title='Embeddings train with UMAP')

# Test
# Compute PCA
test_embeddings_pca = pca.fit_transform(val_features)
plot_embeddings(test_embeddings_pca, val_targets, title='Embeddings test with PCA')
# Compute TSNE
test_embeddings_tsne = TSNE(n_components=2).fit_transform(val_features)
plot_embeddings(test_embeddings_tsne, val_targets, title='Embeddings test with TSNE')
# Compute UMAP
test_embeddings_umap = umap.UMAP().fit_transform(val_features)
plot_embeddings(test_embeddings_umap, val_targets, title='Embeddings test with UMAP')

print('end')


