import glob
from itertools import chain
import os
import random
import zipfile
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import re
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
from transformers import BertTokenizer, BertModel
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
        id = self.ids[id]
        filename = self._load_image(id)
        path = os.path.join(self.data, filename)
        anchor = random.choice(self._load_captions(id)) 
        positive = Image.open(path).convert('RGB')           
        negative = random.choice(self.ids)
        while negative == id:
            negative = random.choice(self.ids)
        filename_negative = self._load_image(negative)
        path_negative = os.path.join(self.data, filename_negative)
        negative = Image.open(path_negative).convert('RGB') 

        if self.transform is not None:
            positive = self.transform(positive)
            negative = self.transform(negative)

        return (anchor, positive, negative)


    def __len__(self) -> int:
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
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#Split validation in two
coco_valid = COCO(INSTANCES_VAL)
ids = list(sorted(coco_valid.imgs.keys()))
np.random.shuffle(ids)
splitted = np.split(np.asarray(ids), 2)
    

# Datasets and Dataloaders
train_data = TripletData(root=PATH_TRAIN, annFile=INSTANCES_TRAIN, captionsFile=CAPTIONS_TRAIN, annotations=ANNOTATIONS, ids_list=splitted[0] transform=train_transforms)
val_data = TripletData(root=PATH_VAL, annFile=INSTANCES_VAL, captionsFile=CAPTIONS_VAL, annotations=ANNOTATIONS, ids_list=splitted[1], transform=val_transforms, train=False)

train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=32, shuffle=True, num_workers=0) #
val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size=32, shuffle=True,  num_workers=0)


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
        self.linear = nn.Linear(self.bert.config.hidden_size, 1000)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]
        caption_embedding = self.linear(cls_embedding)
        return caption_embedding
    

epochs = 2
device = 'cuda'

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Create caption encoder and optimizer
caption_encoder = CaptionEncoder(bert_model)
caption_encoder = caption_encoder.to(device)
# Our base model


model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 512)
model = model.to(device)
optimizer = optim.Adam(list(model.parameters()) + list(caption_encoder.parameters()), lr=1e-3)


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
        anchor = [re.sub(r'[^a-zA-Z ]', '', sentence.lower()) for sentence in anchor]
        input_ids_anchor = tokenizer.encode(anchor, add_special_tokens=True, padding='max_length', truncation=True, max_length=256, return_tensors='pt').to(device)
        attention_mask_anchor = input_ids_anchor.ne(0).to(device)

        embedding_anchor = caption_encoder(input_ids_anchor, attention_mask_anchor)
        embedding_positive = model(positive.to(device))
        embedding_negative = model(negative.to(device))

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
            anchor = [re.sub(r'[^a-zA-Z ]', '', sentence.lower()) for sentence in anchor]
            input_ids_anchor = tokenizer.encode(anchor, add_special_tokens=True, padding='max_length', truncation=True, max_length=256, return_tensors='pt').to(device)
            attention_mask_anchor = input_ids_anchor.ne(0).to(device)

            embedding_anchor = caption_encoder(input_ids_anchor, attention_mask_anchor)
            embedding_positive = model(positive.to(device))
            embedding_negative = model(negative.to(device))
            loss = triplet_loss(embedding_anchor, embedding_positive, embedding_negative)
            index += 1
            val_loss += loss
    print("Val Loss: {}".format(val_loss.item() / index))


   

torch.save(model.state_dict(), "weights-resnet-task-d.pth")
torch.save(caption_encoder.state_dict(), "weights-caption-encoder-task-d.pth")

