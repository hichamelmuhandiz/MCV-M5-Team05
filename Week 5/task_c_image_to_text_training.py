import glob
from itertools import chain
import os
import random
import zipfile
from tqdm.notebook import tqdm
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models, ops
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image
import json
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from pycocotools.coco import COCO
import copy

PATH_TRAIN = "/export/home/mcv/datasets/COCO/train2014/"
PATH_VAL = "/export/home/mcv/datasets/COCO/val2014/"
INSTANCES_TRAIN = "/export/home/mcv/datasets/COCO/instances_train2014.json"
INSTANCES_VAL = "/export/home/mcv/datasets/COCO/instances_val2014.json"
CAPTIONS_TRAIN = "/export/home/mcv/datasets/COCO/captions_train2014.json"
CAPTIONS_VAL = "/export/home/mcv/datasets/COCO/captions_val2014.json"
ANNOTATIONS = "/export/home/mcv/datasets/COCO/mcv_image_retrieval_annotations.json"



class TripletData(datasets.VisionDataset):
    
    def __init__(self, root: str, annFile: str, captionsFile: str, annotations: str, ids_list, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None, train=True) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.data = root


        self.ids = ids_list
        self.coco = COCO(annFile)
        self.coco_captions = COCO(captionsFile)

        self.tuples = []
        for id in self.ids:
            captions = self._load_captions(id)
            for caption in captions:
                self.tuples.append((id, caption))
        self.coco.loadImgs()

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return path

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))
    
    def _load_captions(self, id: int) -> List[Any]:
        return [ann["caption"] for ann in self.coco_captions.loadAnns(self.coco_captions.getAnnIds(id))]

    def __getitem__(self, id: int) -> Tuple[Any, Any]:
        # idx = int(self.tuples[id][0])
        # filename = self._load_image(idx)
        # path = os.path.join(self.data, filename)
        # anchor = Image.open(path).convert('RGB')
        # positive = self.tuples[id][1]             
        # negative = random.choice(self.ids)
        # while negative == idx:
        #     negative = random.choice(self.ids)
        # negative = random.choice(self._load_captions(negative))

        idx = int(self.ids[id])
        filename = self._load_image(idx)
        path = os.path.join(self.data, filename)
        anchor = Image.open(path).convert('RGB')
        positive = random.choice(self._load_captions(idx))
        negative = random.choice(self.ids)
        while negative == idx:
            negative = random.choice(self.ids)
        negative = random.choice(self._load_captions(int(negative)))

        if self.transform is not None:
            anchor = self.transform(anchor)

        return (anchor, positive, negative)


    def __len__(self) -> int:
        # return len(self.tuples)
        return len(self.ids)


# Transforms
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#Split validation in two
coco_valid = COCO(INSTANCES_VAL)
ids = list(sorted(coco_valid.imgs.keys()))
np.random.shuffle(ids)
splitted = np.split(np.asarray(ids), 2)

# Datasets and Dataloaders
train_data = TripletData(root=PATH_VAL, annFile=INSTANCES_VAL, captionsFile=CAPTIONS_VAL, annotations=ANNOTATIONS, ids_list=splitted[0], transform=val_transforms, train=False)
val_data = TripletData(root=PATH_VAL, annFile=INSTANCES_VAL, captionsFile=CAPTIONS_VAL, annotations=ANNOTATIONS, ids_list=splitted[1], transform=val_transforms, train=False)

train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=64, shuffle=True, num_workers=0) #
val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size=64, shuffle=True,  num_workers=0)


class TripletLoss(nn.Module):
    def __init__(self, margin=1):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    # Distances in embedding space is calculated in euclidean
    def forward(self, anchor, positive, negative):
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
    
# Define a simple linear layer to map BERT hidden state to caption embedding
class CaptionEncoder(nn.Module):
    def __init__(self, bert_model):
        super(CaptionEncoder, self).__init__()
        self.bert = bert_model
        self.linear = nn.Sequential(nn.Linear(768, 256),
                                        nn.Dropout(),
                                     nn.ReLU(),
                                nn.Linear(256, 128))
        
        for param in self.linear.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]
        caption_embedding = self.linear(cls_embedding)
        return caption_embedding
    

epochs = 3
device = 'cuda'

# Load pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')

# Create caption encoder and optimizer
caption_encoder = CaptionEncoder(bert_model)
caption_encoder = caption_encoder.to(device)
# Our base model

model = models.resnet50(pretrained=True)

for param in model.parameters():
        param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.Dropout(),
    nn.ReLU(),
    nn.Linear(256, 128))

for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)
optimizer = optim.Adam(list(model.parameters()) + list(caption_encoder.parameters()), lr=1e-2)


triplet_loss = TripletLoss(margin=1)

# Training
for epoch in range(epochs):
    model.train()
    caption_encoder.train()
    epoch_loss = 0.0
    index = 0

    for data in enumerate(train_loader):
        optimizer.zero_grad()
        index, content = data
        anchor,positive,negative = content

        positive = [re.sub(r'[^a-zA-Z ]', '', sentence.lower()) for sentence in positive]
        negative = [re.sub(r'[^a-zA-Z ]', '', sentence.lower()) for sentence in negative]
        print(positive[0])
        print(len(positive))
        positive = tokenizer(positive, return_tensors="pt", padding = True)
        input_ids_positive = positive["input_ids"].to(device)
        attention_mask_positive = positive["attention_mask"].to(device)
        negative = tokenizer(negative, return_tensors="pt", padding = True)
        input_ids_negative = negative["input_ids"].to(device)
        attention_mask_negative = negative["attention_mask"].to(device)
        

        embedding_anchor = model(anchor.to(device))
        embedding_positive = caption_encoder(input_ids_positive, attention_mask_positive)
        print(embedding_positive.shape)
        embedding_negative = caption_encoder(input_ids_negative, attention_mask_negative)
        print(loss.calc_euclidean(embedding_anchor - embedding_positive))
        loss = triplet_loss(embedding_anchor, embedding_positive, embedding_negative)
        index += 1
        epoch_loss += loss
        print("Batch Loss: {}".format(epoch_loss.item() / index))
        loss.backward()
        optimizer.step()
    print("Epoch Loss: {}".format(epoch_loss.item() / index))

    index = 0
    val_loss = 0.0
    with torch.no_grad():
        for data in enumerate(val_loader):
            idx, content = data
            anchor,positive,negative = content

            positive = [re.sub(r'[^a-zA-Z ]', '', sentence.lower()) for sentence in positive]
            negative = [re.sub(r'[^a-zA-Z ]', '', sentence.lower()) for sentence in negative]

            positive = tokenizer(positive, return_tensors="pt", padding = True)
            input_ids_positive = positive["input_ids"].to(device)
            attention_mask_positive = positive["attention_mask"].to(device)

            negative = tokenizer(negative, return_tensors="pt", padding = True)
            input_ids_negative = negative["input_ids"].to(device)
            attention_mask_negative = negative["attention_mask"].to(device)
            
            embedding_anchor = model(anchor.to(device))
            embedding_positive = caption_encoder(input_ids_positive, attention_mask_positive)
            embedding_negative = caption_encoder(input_ids_negative, attention_mask_negative)
            loss = triplet_loss(embedding_anchor, embedding_positive, embedding_negative)
            index += 1
            val_loss += loss
    print("Val Loss: {}".format(val_loss.item() / index))

    torch.save(model.state_dict(), "weights-resnet-definitive.pth")
    torch.save(caption_encoder.state_dict(), "weights-caption-encoder-definitive.pth")



