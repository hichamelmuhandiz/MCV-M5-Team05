import os, sys
import torch
import random
import time
import cv2
import glob
from itertools import combinations

import torch.nn as nn
import torch.optim as optim

from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torch.autograd import Variable
from tqdm import tqdm
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score,precision_score, recall_score, average_precision_score,precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import PrecisionRecallDisplay

from torch.utils.data.sampler import BatchSampler
from pytorch_metric_learning.utils.inference import FaissKNN
import umap.umap_ as umap
#import faiss
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils import accuracy_calculator
import torch.nn.functional as F
from PIL import Image
import wandb

log_wandb = False
if log_wandb:
    writer = wandb.init(
        # Set the project where this run will be logged
        project="w5",name = "150 epochs",
        # Track hyperparameters and run metadata
        config={
            "epochs": 150,
        })

cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
if device.type == "cuda":
    torch.cuda.get_device_name()
class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()
class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)
        
#P1 model
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(238144 , 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)
    
class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(
                'Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}'.format(
                    epoch, batch_idx, loss, mining_func.num_triplets
                )
            )
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

def test(train_set, test_set, model, accuracy_calculator):
    print("Get all train embeddings")
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    print("Get all test embeddings")
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print('Computting accuracy')
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, test_labels, train_embeddings, train_labels, False
    )
    print('Test set accuracy (Precision@1) = {}'.format(accuracies['precision_at_1']))
# Define the data preprocessing pipeline
data_transform = transforms.Compose([transforms.ToTensor()])


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix

class TripletDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        if len(img1.shape) > 2:
            img1 = Image.fromarray(np.uint8(img1.permute(1, 2, 0).numpy() * 255))
            img2 = Image.fromarray(np.uint8(img2.permute(1, 2, 0).numpy() * 255))
            img3 = Image.fromarray(np.uint8(img3.permute(1, 2, 0).numpy() * 255))
        else:
            img1 = Image.fromarray(img1.numpy(), mode='L')
            img2 = Image.fromarray(img2.numpy(), mode='L')
            img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)
class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError
class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
dataset_path = '/home/mcv/datasets/MIT_split'
train_dataset = torchvision.datasets.ImageFolder(root=dataset_path+"/train/", transform=data_transform)
val_dataset = torchvision.datasets.ImageFolder(root=dataset_path+"/test/",  transform=data_transform)
train_dataset.train = True
val_dataset.train = False
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

train_dataset.train_labels = torch.from_numpy(np.array(train_dataset.targets))
val_dataset.test_labels = torch.from_numpy(np.array(val_dataset.targets))

train_dataset.train_data = torch.from_numpy(np.array([s[0].numpy() for s in train_dataset]))
val_dataset.test_data = torch.from_numpy(np.array([s[0].numpy() for s in val_dataset]))


inv_class_to_idx = {v: k for k, v in train_dataset.class_to_idx.items()}

batch_size = 16
train_batch_sampler = BalancedBatchSampler(train_dataset.train_labels, n_classes=8, n_samples=25)
test_batch_sampler = BalancedBatchSampler(val_dataset.test_labels, n_classes=8, n_samples=25)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(val_dataset, batch_sampler=test_batch_sampler, **kwargs)


mnist_classes = train_dataset.classes
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

def plot_embeddings(embeddings, targets, title='', xlim=None, ylim=None, filename="Test"):
    plt.figure(figsize=(10,10))
    plt.title(title)
    for i in range(8):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)
    plt.savefig(str(filename+".png"))

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels

def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None

class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError

class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def RandomNegativeTripletSelector(margin, cpu=False): 
    return FunctionNegativeTripletSelector(margin=margin,negative_selection_fn=random_hard_negative,cpu=cpu)
# Set up the network and training parameters
margin = 1.
embedding_net = EmbeddingNet()
mining = False
if mining:
    model =embedding_net
    loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
else:
    model = TripletNet(embedding_net)
    
    loss_fn = TripletLoss(margin)
    
if log_wandb:wandb.watch(model)
if cuda:
    model.cuda()
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 150
log_interval = 10
class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model
    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        
        if log_wandb: writer.log({"train_loss": train_loss})
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
                                                                                 
        if log_wandb : writer.log({"val_loss": val_loss})
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)
      
def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()

    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics

def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics
class AverageNonzeroTripletsMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'

#mode = "train"
#mode = "plot_metrics"
mode = "retrieval"
if mode == "train":
    if mining:
        fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AverageNonzeroTripletsMetric()])
    else:
                
        triplet_train_dataset = TripletDataset(train_dataset) # Returns pairs of images and target same/different
        triplet_test_dataset = TripletDataset(val_dataset)

        triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True)
        triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False)
        fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)
    torch.save(model.state_dict(), 'triplet_150_epoch2.pth')
elif mode == "plot_metrics":
    epoch_str = "30"
    model.load_state_dict(torch.load('triplet_30_epoch.pth'))  # Load model weights
    model.eval()  # Set model to evaluation mode
    train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
    plot_embeddings(train_embeddings_cl, train_labels_cl, title='Train embeddings', filename = "train_embed"+epoch_str)
    val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
    plot_embeddings(val_embeddings_cl, val_labels_cl, title='Test embeddings', filename = "test_embed"+epoch_str)
    #PCA
    pca = PCA(n_components=2)
    train_embeddings_pca = pca.fit_transform(train_embeddings_cl)
    plot_embeddings(train_embeddings_pca, train_labels_cl, title='Train embeddings with PCA', filename = "train_embed_PCA"+epoch_str)
    test_embeddings_pca = pca.fit_transform(val_embeddings_cl)
    plot_embeddings(test_embeddings_pca, val_labels_cl, title='Test embeddings with PCA', filename = "test_embed_PCA"+epoch_str)

    #TSNE
    train_embeddings_tsne = TSNE(n_components=2).fit_transform(train_embeddings_cl)
    plot_embeddings(train_embeddings_tsne, train_labels_cl, title='Train embeddings with TSNE', filename = "train_embed_TSNE"+epoch_str)
    test_embeddings_tsne = TSNE(n_components=2).fit_transform(val_embeddings_cl)
    plot_embeddings(test_embeddings_tsne, val_labels_cl, title='Test embeddings with TSNE', filename = "test_embed_TSNE"+epoch_str)

    #UMAP
    train_embeddings_umap = umap.UMAP().fit_transform(train_embeddings_cl)
    plot_embeddings(train_embeddings_umap, train_labels_cl, title='Train embeddings with UMAP',filename = "train_embed_UMAP"+epoch_str)
    test_embeddings_umap = umap.UMAP().fit_transform(val_embeddings_cl)
    plot_embeddings(test_embeddings_umap, val_labels_cl, title='Test embeddings with UMAP',filename = "test_embed_UMAP"+epoch_str)


elif mode=="retrieval":
    
    model.load_state_dict(torch.load('triplet_150_epoch.pth'))  # Load model weights
    model.eval()  # Set model to evaluation mode
    
    img_width = 256
    img_height = 256
    train_images = np.zeros((0, 3, img_height, img_width))
    val_images = np.zeros((0, 3, img_height, img_width))
    
    labels_train = []
    labels_val = []
    
    val_features = np.zeros((0, 2))
    with torch.no_grad():
        model.eval()
        train_embeddings = np.zeros((len(train_loader.dataset), 2))
        labels = np.zeros(len(train_loader.dataset))
        k = 0
        for images, target in train_loader:
            train_images = np.concatenate((train_images, images.detach().cpu().numpy()), axis=0)

            if cuda:
                images = images.cuda()
            train_embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            
            
            labels_train.extend(target.detach().cpu().numpy().flatten().tolist())

            k += len(images)
    count = 0
    with torch.no_grad():
        model.eval()
        val_embeddings = np.zeros((len(test_loader.dataset), 2))
        labels = np.zeros(len(test_loader.dataset))
        k = 0
        for images, target in test_loader:
            #print("images ", images)
            val_images = np.concatenate((val_images, images.detach().cpu().numpy()), axis=0)

            #print("img list ",train_images)
            if cuda:
                images = images.cuda()
            val_embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            
            outputs= model.get_embedding(images).data.cpu().numpy()
            
            n_samples = outputs.shape[0]
            outputs_flat = outputs.reshape(n_samples, outputs.shape[1])
            val_features = np.concatenate((val_features, outputs_flat), axis=0)
            labels_val.extend(target.detach().cpu().numpy().flatten().tolist())
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
            count = count +1 
    k = 10 # number of neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    labels_train = np.array(labels_train)
    labels_val = np.array(labels_val)

    val_targets = labels_val
    train_targets = labels_train

    knn.fit(train_embeddings,train_targets)
    distances, indices = knn.kneighbors(val_embeddings)
    

    print('Retrieved images:', indices)

    
    class_names = train_dataset.classes
    fig, ax = plt.subplots(4, 6, figsize=(10, 10), dpi=200)
    for i in range (0,4):
        ax[i][0].imshow(np.moveaxis(val_images[i], 0, -1))
        ax[i][0].set_xticks([])
        ax[i][0].set_yticks([])
        class_name = class_names[labels_val[i]]
        ax[i][0].set_title("Test img:\n" + class_name, fontsize=9)
        for j in range (0,5):
            ax[i][j+1].imshow(np.moveaxis(train_images[indices[i,j]], 0, -1))
            ax[i][j+1].set_xticks([])
            ax[i][j+1].set_yticks([])
            class_name = class_names[labels_train[indices[i][j]]]
            ax[i][j+1].set_title("Retrieved img:\n" + class_name, fontsize=9)
    fig.tight_layout()
    plt.savefig('retrieval_cnew2.png')
    plt.close()

    # Compute the evaluation metrics
    APs = []
    Precisions_at_1 = []
    Precisions_at_5 = []
    #val_features = val_embeddings
    train_features = train_embeddings
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

    # Compute Precision Recall curve for Image Retrieval
    val_prob = knn.predict_proba(val_embeddings)
    fig, ax = plt.subplots(1, 1, figsize=(10,10), dpi=150)
    ax.set_title("Precision-Recall curve", size=16)
    for class_id, class_name in enumerate(class_names):
        PrecisionRecallDisplay.from_predictions(np.where(val_targets==class_id, 1, 0),
                                                val_prob[:, class_id],
                                                ax=ax, name=class_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.savefig("pr_curve_c_cool.png")
    #####
    fig, ax = plt.subplots(1, 1, figsize=(10,10), dpi=200)
    ax.set_title("Precision-Recall curve", size=16)
    for class_id, class_name in enumerate(class_names):
        PrecisionRecallDisplay.from_predictions(np.where(val_targets==class_id, 1, 0),
                                                np.where(train_targets[indices][:, 0]==class_id, 1, 0),
                                                ax=ax, name=class_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.savefig('pr_c.png')
    sys.exit()