import argparse
import torch
from tqdm import tqdm
import os
from models import mlp
from data.dataset import data_loader as loader
from models.mlp import Classifier


#python mytest.py --vit_dir /home/thao.nguyen/AI701B/SEViT/models/X-ray/m_best_model.pth  --root_dir /home/thao.nguyen/AI701B/SEViT/data/X-ray
#thao.nguyen

image_size = (224,224)
batch_size = 32


parser = argparse.ArgumentParser(description='Testing ViT')
parser.add_argument('--vit_dir', type=str , help='pass the path of downloaded ViT')
parser.add_argument('--root_dir', type=str, help='pass the path of downloaded data')
args = parser.parse_args()

root_dir=args.root_dir
vit_dir=args.vit_dir

data_loader, image_dataset = loader(root_dir=root_dir, batch_size= batch_size, image_size=image_size)
test=data_loader['test']

model = torch.load(vit_dir).cuda()
model.eval()

test_acc = 0.0
total=0
test=data_loader['test']
for images, labels in tqdm(test): 
    images = images.cuda()
    labels= labels.cuda()
    with torch.no_grad(): 
        model.eval()
        output = model(images)
        prediction = torch.argmax(output, dim=-1)
        test_acc += sum(prediction == labels).float().item()
        total += len(labels)

print(f'Testing accuracy = {(test_acc/total):.4f}')