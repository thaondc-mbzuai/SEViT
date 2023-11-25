import os 
import torch 
import argparse
import numpy as np

from utils import *
# from utils import get_b0_list
from data.dataset import data_loader, data_loader_attacks

#from models import mlp

def get_b0_list(MLP_path, num_classifiers=5): 
    """
    Return list of intermdiate MLPs. 
    
    Args: 
        MLP_path: Path of the downloaded MLPs directory.
    
    """
    i=0
    classifiers_list = [0]*num_classifiers
    for classif in sorted(os.listdir(MLP_path)): 
        classifiers_list[i] = torch.load(os.path.join(MLP_path, classif)).eval().cuda() # When saving and loading model like these, please remember that we have 
        i+=1
        print(f'MLP {i} is loaded!')
    return classifiers_list


def get_classifiers_list(MLP_path, num_classifiers=5): 
    """
    Return list of intermdiate MLPs. 
    
    Args: 
        MLP_path: Path of the downloaded MLPs directory.
    
    """
    i=0
    classifiers_list = [0]*num_classifiers
    for classif in sorted(os.listdir(MLP_path)): 
        classifiers_list[i] = torch.load(os.path.join(MLP_path, classif)).eval().cuda()
        i+=1
        print(f'MLP {i} is loaded!')
    return classifiers_list

def majority_voting(data_loader, model, mlps_list):
    """
    SEViT performance with majority voting. 

    Args: 
    data_loader: loader of test samples for clean images, or attackes generated from the test samples
    model: ViT model 
    mlps_list: list of intermediate MLPs

    Return: 
    Accuracy. 

    """
    acc_ = 0.0 
    for images, labels in data_loader:
        final_prediction = []
        images = images.cuda()
        vit_output = model(images)
        vit_predictions = torch.argmax(vit_output.detach().cpu(), dim=-1)
        final_prediction.append(vit_predictions.detach().cpu())
        for mlp in mlps_list:
            mlp_output = mlp(images)
            mlp_predictions = torch.argmax(mlp_output.detach().cpu(), dim=-1)
            final_prediction.append(mlp_predictions.detach().cpu())
        stacked_tesnor = torch.stack(final_prediction,dim=1)
        preds_major = torch.argmax(torch.nn.functional.one_hot(stacked_tesnor).sum(dim=1), dim=-1)
        acc = (preds_major == labels).sum().item()/len(labels)
        acc_ += acc
    final_acc = acc_ / len(data_loader)
    print(f'Final Accuracy From Majority Voting = {(final_acc *100) :.3f}%' )
    return final_acc

"""
python majority_voting_thao.py --vit_dir /home/thao.nguyen/AI701B/SEViT/models/X-ray/m_best_model.pth \
                            --root_dir  /home/thao.nguyen/AI701B/SEViT/data/X-ray \
                            --mlp_dir /home/thao.nguyen/AI701B/SEViT/models/X-ray/MLP \
                            --attack_list "CW" "PGD" \
                            --images_type "adversarial" \
                            --clf_type "MLP" \
"""
#thao.nguyen


parser = argparse.ArgumentParser(description='Testing ViT')
parser.add_argument('--vit_dir', type=str , help='pass the path of downloaded ViT')
parser.add_argument('--root_dir', type=str, help='pass the path of downloaded data')
parser.add_argument('--mlp_dir', type=str, help='pass the path of downloaded data')
parser.add_argument('--attack_list', type=str , nargs='+', help='Attack List to Generate')
parser.add_argument('--images_type', type=str , choices=['clean', 'adversarial'],
                    help='Path to root directory of images')
parser.add_argument('--clf_type', type=str , choices=['b0', 'MLP'],
                    help='Path to root directory of images')
args = parser.parse_args()

vit_dir= args.vit_dir
root_dir = args.root_dir
mlp_dir = args.mlp_dir

model = torch.load(vit_dir).cuda()
model.eval()
print('ViT is loaded!')
if args.clf_type=='MLP':
    MLPs_list = get_classifiers_list(mlp_dir)
else:
    MLPs_list = get_b0_list(mlp_dir)


if args.images_type=='clean':
    print(root_dir)
    loader_, dataset_ = data_loader(root_dir=root_dir,batch_size=15)
    loader_=loader_['test']
    majority_voting(data_loader=loader_, model= model, mlps_list=MLPs_list)
else:
    attack_names = args.attack_list
    #'PGD','LinfBIM','L2PGD','FGSM','CW','BIM','AUTOPGD'
    for attack_name in attack_names:
        print(f"***{attack_name}***")
        loader_, dataset_ = data_loader_attacks(root_dir='/l/users/xiwei.liu/epsilon0.01/attack_test/', attack_name= attack_name, batch_size=15)
        majority_voting(data_loader=loader_, model= model, mlps_list=MLPs_list)
