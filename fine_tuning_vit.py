from data.dataset import data_loader as loader
from data.dataset import data_loader_attacks

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torchvision.models 
from torchvision.utils import save_image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from sklearn.metrics import auc
import math
from matplotlib.backends.backend_pdf import PdfPages
import os
import timm
import sys
from tqdm import tqdm
import pickle
import argparse

#python fine_tuning_vit.py --output_dir /home/thao.nguyen/AI701B/SEViT/models/X-ray  --root_dir /home/thao.nguyen/AI701B/SEViT/data/X-ray
#thao.nguyen

epochs = 150
lr = 1e-8
image_size = (224,224)
batch_size = 32
device='cuda'

parser = argparse.ArgumentParser(description='Finetuning ViT')
parser.add_argument('--output_dir', type=str , help='pass the path of output ViT')
parser.add_argument('--root_dir', type=str, help='pass the path of downloaded data')
args = parser.parse_args()

root_dir = args.root_dir
output_dir = args.output_dir

data_loader, image_dataset = loader(root_dir=root_dir, batch_size= batch_size, image_size=image_size)
train_loader = data_loader['train']
val_loader = data_loader['valid']

model = timm.models.vit_base_patch16_224_in21k(pretrained=True)
model.head = nn.Linear(model.head.in_features, 2)
model=model.cuda()

# model = torch.load('/home/thao.nguyen/AI701B/SEVIT/models/X-ray/m_best_model_9598.pth').cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lr,betas=(0.9, 0.99))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)

best_accuracy = 0.0
for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    model.train()
    
    for (images, labels) in tqdm(train_loader, desc = 'Iterating over train data, Epoch: {}/{}'.format(epoch + 1, epochs+1)):
        # get the inputs
        images = images.cuda()
        labels = labels.cuda()
        # labels = labels.squeeze()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    #accuracy = testAccuracy(val_loader, model, device)
    model.eval()
    accuracy = 0.0
    total = 0.0
    with torch.no_grad():
        for (images, labels) in tqdm(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.squeeze()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    accuracy=accuracy/total
    # wandb.log({"Val_acc": accuracy, "train_loss": running_loss / len(train_ds)})
    # we want to save the model if the accuracy is the best
    if accuracy > best_accuracy:
        best_accuracy=accuracy
        path = os.path.join(output_dir, "m_best_model.pth")
        torch.save(model, path)
        print(f"Best accuracy is updated to ... {accuracy}")
