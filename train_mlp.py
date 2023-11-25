import torch
from torch import nn
from torch.optim import Adam
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import DataLoader
from models import mlp
from models.mlp import Classifier
from tqdm import tqdm
import os
from data.dataset import data_loader
from data import dataset
import argparse
import os
import json
import pickle
# import hydra
# from omegaconf import DictConfig, OmegaConf
# from hydra.utils import get_original_cwd, to_absolute_path
import logging
log = logging.getLogger(__name__)
# import wandb
import numpy as np
from tqdm import tqdm
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES']= '4'
import torch.nn as nn

#python train_mlp.py --vit_dir /home/thao.nguyen/AI701B/SEViT/models/X-ray/m_best_model.pth  --root_dir /home/thao.nguyen/AI701B/SEViT/data/X-ray --output_dir /home/thao.nguyen/AI701B/SEViT/models/X-ray/MLP
#thao.nguyen

parser = argparse.ArgumentParser(description='Training MLPs')
parser.add_argument('--vit_dir', type=str , help='pass the path of downloaded ViT')
parser.add_argument('--root_dir', type=str, help='pass the path of downloaded data')
parser.add_argument('--output_dir', type=str, help='pass the path of output')
args = parser.parse_args()

root_dir=args.root_dir
vit_dir=args.vit_dir
output_dir=args.output_dir

image_size = (224,224)
batch_size = 32
epochs = 70
lr =  1e-5

vit = torch.load(vit_dir).cuda()
vit.eval()
for w in vit.parameters(): 
    w.requires_grad = False

data_loader, image_dataset = data_loader(root_dir=root_dir, batch_size= batch_size, image_size=image_size)


for index in range(5):
    print(f'***Block {index+1}***')

    classifier = Classifier(num_classes=2,vit=vit, block_num=index).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr = lr,betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)
    #scheduler = StepLR(optimizer=optimizer, step_size=15, gamma=0.1, verbose=True)

    classifier.train()
    best_accuracy=0

    for epoch in range(epochs): 
        #Training
        for images, labels in tqdm(data_loader['train'],desc = 'Epoch: {}/{}'.format(epoch + 1, epochs)):
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
        scheduler.step() 
        #Validation 
        classifier.eval()
        accuracy = 0
        total = 0

        with torch.no_grad(): 
            for images, labels in tqdm(data_loader['test']):
                images = images.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()
                outputs = classifier(images)
                loss = criterion(outputs, labels)

                prediction = torch.argmax(outputs , dim= -1)
                total += labels.size(0)
                accuracy += (prediction == labels).sum().item()

        accuracy=accuracy/total
        if accuracy > best_accuracy:
            best_accuracy=accuracy
            path = os.path.join(output_dir, f"m_best_model_mlp_block_{index}.pth")    
            torch.save(classifier, path)
    print(f"Best accuracy {index+1} is updated to ... {best_accuracy}")
    